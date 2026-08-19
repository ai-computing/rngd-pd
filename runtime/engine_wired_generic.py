"""Phase 4 — CONFIG-DRIVEN resident-KV wired engine (Llama/Qwen2 dense family, one code path).

Generalizes llama_engine_resident.py: all dims come from ModelConfig (hidden, inter, n_heads, n_kv,
head_dim, gqa), DSL kernels are generated per-dim (dsl_gen.py), and optional QKV bias / tied embed /
llama3-vs-plain RoPE are handled — so the SAME fast wired path serves Llama-3.1-8B AND Qwen2.5-0.5B
(and other dense 8B/70B) by config alone. Keeps the resident-KV feedback (no per-token KV upload).

Bias handling: q is pre-scaled by 1/sqrt(hd) for the softmax-scale trick, so q-bias is pre-scaled too
(bq*SCALE); k/v bias added as-is; RoPE (a rotation) commutes with the scalar. Bias inputs are always
present (zeros if the model has none) so the compiled graph shape is model-count-invariant.

  python3 engine_wired_generic.py --model Qwen/Qwen2.5-0.5B-Instruct --n 4
  python3 engine_wired_generic.py --model meta-llama/Llama-3.1-8B-Instruct --n 4
"""
import torch, furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
import sys, math, time, argparse, numpy as np
import _paths  # noqa: E402  (portable path bootstrap)
try: import _tv_patch  # noqa
except Exception: pass
from furiosa.torch.custom_ops import TacticKernelModule, CompileModule
from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig
import dsl_gen
from generic_runtime import ModelConfig, GenericRuntime, rmsnorm

DEV = "rngd:0"; BF = torch.bfloat16


class WireLayer(torch.nn.Module):
    def __init__(self, dims, S):
        super().__init__()
        self.H, self.M, self.NH, self.NKV, self.HD, self.G = dims
        self.S = S
        self.rms1 = TacticKernelModule(dsl_gen.rms_norm_dsl(self.H))
        self.rms2 = TacticKernelModule(dsl_gen.rms_norm_dsl(self.H))
        self.sm = TacticKernelModule(dsl_gen.softmax_dsl(self.NKV, self.G, S))
        self.r1 = TacticKernelModule(dsl_gen.zip_add_dsl(self.H))
        self.r2 = TacticKernelModule(dsl_gen.zip_add_dsl(self.H))
        self.sw = TacticKernelModule(dsl_gen.swiglu_dsl(self.M))

    def _rope(self, x, nh, cos, sin):
        HD = self.HD; half = HD//2; xr = x.view(nh, HD); a, b = xr[..., :half], xr[..., half:]
        return torch.cat([a*cos - b*sin, b*cos + a*sin], -1).reshape(nh*HD)

    def forward(self, x, eps, nh, n1, wq, bq, wk, bk, wv, bv, wo, n2, wgu, wd, Kp, Vp, cos, sin, mask, oh):
        NH, NKV, HD, G, M = self.NH, self.NKV, self.HD, self.G, self.M
        h = self.rms1(x, n1, eps, nh)[0]
        q = self._rope(torch.matmul(wq, h) + bq, NH, cos, sin)      # wq,bq pre-scaled by 1/sqrt(hd)
        k = self._rope(torch.matmul(wk, h) + bk, NKV, cos, sin)
        v = torch.matmul(wv, h) + bv
        Kf = Kp + k.view(NKV, HD, 1) * oh.view(1, 1, self.S)
        Vf = Vp + v.view(NKV, 1, HD) * oh.view(1, self.S, 1)
        sc = torch.matmul(q.reshape(NKV, G, HD), Kf) + mask.view(1, 1, self.S)
        p = self.sm(sc.to(BF))[0]
        ao = torch.matmul(p, Vf).reshape(NH*HD)
        x2 = self.r1(torch.matmul(wo, ao), x)[0]
        h2 = self.rms2(x2, n2, eps, nh)[0]
        gg = torch.matmul(wgu[:M], h2); uu = torch.matmul(wgu[M:], h2)
        return self.r2(torch.matmul(wd, self.sw(gg, uu)[0]), x2)[0], Kf, Vf


class WireModel(torch.nn.Module):
    def __init__(self, dims, S, nl):
        super().__init__(); self.nl = nl
        self.layers = torch.nn.ModuleList([WireLayer(dims, S) for _ in range(nl)])
    def forward(self, x, eps, nh, an1, aq, abq, ak, abk, av, abv, ao, an2, awgu, awd, Kp, Vp, cos, sin, mask, oh):
        ks = []; vs = []
        for i in range(self.nl):
            x, Kf, Vf = self.layers[i](x, eps, nh, an1[i], aq[i], abq[i], ak[i], abk[i], av[i], abv[i],
                                       ao[i], an2[i], awgu[i], awd[i], Kp[i], Vp[i], cos, sin, mask, oh)
            ks.append(Kf); vs.append(Vf)
        return x, torch.stack(ks), torch.stack(vs)


def _d(a): return torch.from_numpy(np.ascontiguousarray(a)).to(BF).to(DEV)
def _df(a): return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32).to(DEV)


class GenericWiredRuntime:
    def __init__(self, cfg, sd, S):
        self.c = cfg; self.S = S; self.nl = cfg.n_layers; self.inv = cfg.inv_freq()
        H, M = cfg.hidden, cfg.inter; NH, NKV, HD, G = cfg.n_heads, cfg.n_kv, cfg.head_dim, cfg.gqa
        self.dims = (H, M, NH, NKV, HD, G); self.SCALE = 1.0/math.sqrt(HD)
        nl = self.nl; g = lambda k: sd[k].astype(np.float32); sqrtH = math.sqrt(H)
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        has_bias = "model.layers.0.self_attn.q_proj.bias" in sd
        zbq = np.zeros(NH*HD, np.float32); zkv = np.zeros(NKV*HD, np.float32)
        gb = lambda k, z: (g(k) if k in sd else z)
        an1 = torch.stack([_d(g(f"model.layers.{i}.input_layernorm.weight")*sqrtH) for i in range(nl)])
        an2 = torch.stack([_d(g(f"model.layers.{i}.post_attention_layernorm.weight")*sqrtH) for i in range(nl)])
        aq = torch.stack([_d(g(f"model.layers.{i}.self_attn.q_proj.weight")*self.SCALE) for i in range(nl)])
        abq = torch.stack([_d(gb(f"model.layers.{i}.self_attn.q_proj.bias", zbq)*self.SCALE) for i in range(nl)])
        ak = torch.stack([_d(g(f"model.layers.{i}.self_attn.k_proj.weight")) for i in range(nl)])
        abk = torch.stack([_d(gb(f"model.layers.{i}.self_attn.k_proj.bias", zkv)) for i in range(nl)])
        av = torch.stack([_d(g(f"model.layers.{i}.self_attn.v_proj.weight")) for i in range(nl)])
        abv = torch.stack([_d(gb(f"model.layers.{i}.self_attn.v_proj.bias", zkv)) for i in range(nl)])
        ao = torch.stack([_d(g(f"model.layers.{i}.self_attn.o_proj.weight")) for i in range(nl)])
        awgu = torch.stack([_d(np.concatenate([g(f"model.layers.{i}.mlp.gate_proj.weight"), g(f"model.layers.{i}.mlp.up_proj.weight")], 0)) for i in range(nl)])
        awd = torch.stack([_d(g(f"model.layers.{i}.mlp.down_proj.weight")) for i in range(nl)])
        self.W = (an1, aq, abq, ak, abk, av, abv, ao, an2, awgu, awd)
        self.eps_d = _df(np.full(H, cfg.eps*H, np.float32)); self.nh_d = _df(np.full(H, -0.5, np.float32))
        self.bias_present = has_bias
        half = HD//2
        z = lambda *s: torch.zeros(*s, dtype=BF)
        sample = (z(H), torch.zeros(H, dtype=torch.float32), torch.zeros(H, dtype=torch.float32),
                  z(nl, H), z(nl, NH*HD, H), z(nl, NH*HD), z(nl, NKV*HD, H), z(nl, NKV*HD),
                  z(nl, NKV*HD, H), z(nl, NKV*HD), z(nl, H, NH*HD), z(nl, H), z(nl, 2*M, H), z(nl, H, M),
                  z(nl, NKV, HD, S), z(nl, NKV, S, HD), z(half), z(half), z(S), z(S))
        cfg_c = FCConfig(allow_external_operators=True, allow_unlowered_operators=True, tactic_hint=TacticHintConfig.ForLlmModelDecode)
        t0 = time.time()
        self.m = CompileModule.from_module(WireModel(self.dims, S, nl), sample, compiler_config=cfg_c); self.m.to(DEV)
        self.compile_s = time.time()-t0

    def argmax_per_pos(self, ids, timeit=False):
        S = self.S; nl = self.nl; NKV, HD = self.c.n_kv, self.c.head_dim
        an1, aq, abq, ak, abk, av, abv, ao, an2, awgu, awd = self.W
        Kp_dev = _d(np.zeros((nl, NKV, HD, S), np.float32))
        Vp_dev = _d(np.zeros((nl, NKV, S, HD), np.float32))
        out = []; tt = 0.0
        for pos, t in enumerate(ids):
            ang = pos*self.inv; cos = _d(np.cos(ang)); sin = _d(np.sin(ang))
            mask = np.full(S, -1e4, np.float32); mask[:pos+1] = 0.0
            oh = np.zeros(S, np.float32); oh[pos] = 1.0
            x = _d(self.embed[t].astype(np.float32))
            t0 = time.time()
            hid, Kp_dev, Vp_dev = self.m(x, self.eps_d, self.nh_d, an1, aq, abq, ak, abk, av, abv, ao,
                                         an2, awgu, awd, Kp_dev, Vp_dev, cos, sin, _d(mask), _d(oh))
            if timeit: tt += time.time()-t0
            out.append(int(np.argmax(self.lm @ rmsnorm(hid.detach().float().cpu().numpy(), self.norm, self.c.eps))))
        self.step_ms = tt/len(ids)*1000 if timeit else 0
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is"); ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    from generic_runtime_wired import _bucket
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd); seq = ids + oracle.generate(ids, a.n); oam = oracle.argmax_per_pos(seq)
    print(f"[generic-wired] {a.model}: H={cfg.hidden} M={cfg.inter} heads={cfg.n_heads}/{cfg.n_kv} hd={cfg.head_dim} "
          f"L={cfg.n_layers} tied={cfg.tie} llama3_rope={'Y' if cfg.rope_scaling else 'N'}", flush=True)
    rt = GenericWiredRuntime(cfg, sd, _bucket(len(seq)))
    print(f"  bias={rt.bias_present} | compiled in {rt.compile_s:.0f}s", flush=True)
    fam = rt.argmax_per_pos(seq, timeit=True)
    ok = sum(f == o for f, o in zip(fam, oam))
    print("\n" + "=" * 64)
    print(f"CONFIG-DRIVEN resident-KV wired — {a.model}")
    print(f"  argmax {ok}/{len(seq)} {'PASS ✅' if ok==len(seq) else 'CHECK ❌'} | {rt.step_ms:.1f} ms/token | compile {rt.compile_s:.0f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
