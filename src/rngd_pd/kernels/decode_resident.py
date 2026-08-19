"""Phase 3 — RESIDENT-KV whole-model wired Llama-8B (removes per-token KV re-upload).

llama_engine_fullwire.py re-uploads the entire KV cache (Kp[NL,NKV,HD,S], Vp[NL,NKV,S,HD] ≈ 16.8MB×2)
from host to device EVERY token (via _d(Kp)) and downloads the new k/v rows back — the reason it was
48.2ms/token vs the 11.86ms perf harness. Here each layer RETURNS the injected full cache Kf/Vf and
the runtime keeps them RESIDENT on device, feeding them back device→device with no host round-trip
(validated bit-exact by probe_resident_kv.py). Same fast+correct wire, same real weights.

  python3 llama_engine_resident.py --model meta-llama/Llama-3.1-8B-Instruct --n 4
"""
import torch, furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
import sys, math, time, argparse, numpy as np
try: import _tv_patch  # noqa
except Exception: pass
from furiosa.torch.custom_ops import TacticKernelModule, CompileModule
from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig
from rngd_pd.kernels import dsl_b1 as LE
from rngd_pd.engines.model import ModelConfig, GenericRuntime, rmsnorm

DEV = "rngd:0"; BF = torch.bfloat16
H, M, NH, NKV, HD, G = 4096, 14336, 32, 8, 128, 4
EPS = 1e-5; SCALE = 1.0/math.sqrt(HD)


class WireLayer(torch.nn.Module):
    """One layer; RETURNS the injected full cache Kf/Vf (not just the new row) for resident feedback."""
    def __init__(self, S):
        super().__init__(); self.S = S
        self.rms1 = TacticKernelModule(LE.RMS_NORM); self.rms2 = TacticKernelModule(LE.RMS_NORM)
        self.sm = TacticKernelModule(LE.softmax_dsl(S))
        self.r1 = TacticKernelModule(LE.ZIP_ADD); self.r2 = TacticKernelModule(LE.ZIP_ADD)
        self.sw = TacticKernelModule(LE.SWIGLU)
    def _rope(self, x, nh, cos, sin):
        half = HD//2; xr = x.view(nh, HD); a, b = xr[..., :half], xr[..., half:]
        return torch.cat([a*cos - b*sin, b*cos + a*sin], -1).reshape(nh*HD)
    def forward(self, x, eps, nh, n1, wq, wk, wv, wo, n2, wgu, wd, Kp, Vp, cos, sin, mask, oh):
        S = self.S
        h = self.rms1(x, n1, eps, nh)[0]
        q = self._rope(torch.matmul(wq, h), NH, cos, sin)
        k = self._rope(torch.matmul(wk, h), NKV, cos, sin); v = torch.matmul(wv, h)
        Kf = Kp + k.view(NKV, HD, 1) * oh.view(1, 1, S)     # inject current row at pos (onehot)
        Vf = Vp + v.view(NKV, 1, HD) * oh.view(1, S, 1)
        sc = torch.matmul(q.reshape(NKV, G, HD), Kf) + mask.view(1, 1, S)
        p = self.sm(sc.to(BF))[0]
        ao = torch.matmul(p, Vf).reshape(NH*HD)
        x2 = self.r1(torch.matmul(wo, ao), x)[0]
        h2 = self.rms2(x2, n2, eps, nh)[0]
        gg = torch.matmul(wgu[:M], h2); uu = torch.matmul(wgu[M:], h2)
        return self.r2(torch.matmul(wd, self.sw(gg, uu)[0]), x2)[0], Kf, Vf   # <-- return full caches


class WireModel(torch.nn.Module):
    def __init__(self, S, nl):
        super().__init__(); self.S = S; self.nl = nl
        self.layers = torch.nn.ModuleList([WireLayer(S) for _ in range(nl)])
    def forward(self, x, eps, nh, an1, aq, ak_, av_, ao_, an2, awgu, awd, Kp, Vp, cos, sin, mask, oh):
        ks = []; vs = []
        for i in range(self.nl):
            x, Kf, Vf = self.layers[i](x, eps, nh, an1[i], aq[i], ak_[i], av_[i], ao_[i], an2[i],
                                       awgu[i], awd[i], Kp[i], Vp[i], cos, sin, mask, oh)
            ks.append(Kf); vs.append(Vf)
        return x, torch.stack(ks), torch.stack(vs)          # updated full caches, stacked over layers


def _d(a): return torch.from_numpy(np.ascontiguousarray(a)).to(BF).to(DEV)
def _df(a): return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32).to(DEV)


class ResidentRuntime:
    def __init__(self, cfg, sd, S, nl):
        self.c = cfg; self.S = S; self.nl = nl; self.inv = cfg.inv_freq()
        g = lambda k: sd[k].astype(np.float32); sqrtH = math.sqrt(H)
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        gg = lambda k: _d(g(k))
        an1 = torch.stack([_d(g(f"model.layers.{i}.input_layernorm.weight")*sqrtH) for i in range(nl)])
        an2 = torch.stack([_d(g(f"model.layers.{i}.post_attention_layernorm.weight")*sqrtH) for i in range(nl)])
        aq = torch.stack([_d(g(f"model.layers.{i}.self_attn.q_proj.weight")*SCALE) for i in range(nl)])
        ak = torch.stack([gg(f"model.layers.{i}.self_attn.k_proj.weight") for i in range(nl)])
        av = torch.stack([gg(f"model.layers.{i}.self_attn.v_proj.weight") for i in range(nl)])
        ao = torch.stack([gg(f"model.layers.{i}.self_attn.o_proj.weight") for i in range(nl)])
        awgu = torch.stack([_d(np.concatenate([g(f"model.layers.{i}.mlp.gate_proj.weight"), g(f"model.layers.{i}.mlp.up_proj.weight")], 0)) for i in range(nl)])
        awd = torch.stack([gg(f"model.layers.{i}.mlp.down_proj.weight") for i in range(nl)])
        self.W = (an1, aq, ak, av, ao, an2, awgu, awd)
        self.eps_d = _df(np.full(H, EPS*H, np.float32)); self.nh_d = _df(np.full(H, -0.5, np.float32))
        half = HD//2
        z = lambda *s: torch.zeros(*s, dtype=BF)
        sample = (z(H), torch.zeros(H, dtype=torch.float32), torch.zeros(H, dtype=torch.float32),
                  z(nl, H), z(nl, NH*HD, H), z(nl, NKV*HD, H), z(nl, NKV*HD, H), z(nl, H, NH*HD),
                  z(nl, H), z(nl, 2*M, H), z(nl, H, M), z(nl, NKV, HD, S), z(nl, NKV, S, HD),
                  z(half), z(half), z(S), z(S))
        cfg_c = FCConfig(allow_external_operators=True, allow_unlowered_operators=True, tactic_hint=TacticHintConfig.ForLlmModelDecode)
        t0 = time.time()
        self.m = CompileModule.from_module(WireModel(S, nl), sample, compiler_config=cfg_c); self.m.to(DEV)
        self.compile_s = time.time()-t0

    def argmax_per_pos(self, ids, timeit=False):
        S = self.S; nl = self.nl
        an1, aq, ak, av, ao, an2, awgu, awd = self.W
        # RESIDENT KV — uploaded ONCE, then fed back device→device (no per-token re-upload)
        Kp_dev = _d(np.zeros((nl, NKV, HD, S), np.float32))
        Vp_dev = _d(np.zeros((nl, NKV, S, HD), np.float32))
        out = []; tt = 0.0
        for pos, t in enumerate(ids):
            ang = pos*self.inv; cos = _d(np.cos(ang)); sin = _d(np.sin(ang))
            mask = np.full(S, -1e4, np.float32); mask[:pos+1] = 0.0
            oh = np.zeros(S, np.float32); oh[pos] = 1.0
            x = _d(self.embed[t].astype(np.float32))
            t0 = time.time()
            hid, Kp_dev, Vp_dev = self.m(x, self.eps_d, self.nh_d, an1, aq, ak, av, ao, an2, awgu, awd,
                                         Kp_dev, Vp_dev, cos, sin, _d(mask), _d(oh))  # feed back, no _d(Kp)
            if timeit: tt += time.time()-t0
            out.append(int(np.argmax(self.lm @ rmsnorm(hid.detach().cpu().float().numpy(), self.norm, self.c.eps))))
        self.step_ms = tt/len(ids)*1000 if timeit else 0
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is"); ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    from rngd_pd.engines.model_wired import _bucket
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd); seq = ids + oracle.generate(ids, a.n); oam = oracle.argmax_per_pos(seq)
    print(f"[resident] compiling WHOLE 32L resident-KV model as ONE EDF...", flush=True)
    rt = ResidentRuntime(cfg, sd, _bucket(len(seq)), cfg.n_layers)
    print(f"  compiled in {rt.compile_s:.0f}s", flush=True)
    fam = rt.argmax_per_pos(seq, timeit=True)
    ok = sum(f == o for f, o in zip(fam, oam))
    print("\n" + "=" * 62)
    print(f"RESIDENT-KV whole-model wired Llama-8B (1 EDF, real weights)")
    print(f"  argmax {ok}/{len(seq)} {'PASS ✅' if ok==len(seq) else 'CHECK'} | {rt.step_ms:.1f} ms/token (was 48.2 w/ per-token KV upload)")
    print("=" * 62)


if __name__ == "__main__":
    main()
