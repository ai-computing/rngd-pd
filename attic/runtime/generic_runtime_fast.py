"""M6 culmination — all dim-general SDK optimizations combined, verified.

Stacks: fusion(8)+mesh(2) [SDK native-TP, full-HBM resident] + resident weights +
weight-concat QKV + FUSED MLP block (gate+up+silu+mul+down as ONE meshed launch, silu/mul
in-graph). NPU launches/layer: 7 (baseline) → 3 (QKV, O, MLP-block). Attention still host
(frontier = RngdStaticCache / NPU paged-attention). Verified vs numpy oracle.

  python generic_runtime_fast.py --model meta-llama/Llama-3.1-8B-Instruct --n 3
"""
import argparse, math, time, numpy as np, torch, sys
import _paths  # noqa: E402  (portable path bootstrap)
torch.set_grad_enabled(False)
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))     # SDK native-TP + full-HBM resident
from furiosa.torch.custom_ops import CompileModule
from generic_runtime import ModelConfig, rmsnorm, rope, GenericRuntime

DEV = "rngd:0"


class QKV(torch.nn.Module):            # weight-as-input concat QKV matmul
    def forward(self, w, x): return w @ x


class MLPBlock(torch.nn.Module):       # fused: down(silu(gate)*up), one launch
    def __init__(self, inter): super().__init__(); self.inter = inter
    def forward(self, wgu, wd, x):
        gu = wgu @ x
        g, u = gu[:self.inter], gu[self.inter:]
        return wd @ (torch.nn.functional.silu(g) * u)


def _dev(a): return torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16).to(DEV)


class FastRuntime:
    def __init__(self, cfg, sd):
        self.c = cfg; self.inv = cfg.inv_freq()
        g = lambda k: sd[k].astype(np.float32)
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        hd = cfg.head_dim; H = cfg.hidden
        self.Hq = cfg.n_heads*hd; self.Hkv = cfg.n_kv*hd; self.qkv_out = self.Hq + 2*self.Hkv
        # compile the 3 block modules once (reused every layer)
        self.qkv_mm = CompileModule.from_module(QKV(), (torch.zeros(self.qkv_out, H, dtype=torch.bfloat16), torch.zeros(H, dtype=torch.bfloat16))); self.qkv_mm.to(DEV)
        self.o_mm = CompileModule.from_module(QKV(), (torch.zeros(H, self.Hq, dtype=torch.bfloat16), torch.zeros(self.Hq, dtype=torch.bfloat16))); self.o_mm.to(DEV)
        self.mlp = CompileModule.from_module(MLPBlock(cfg.inter), (torch.zeros(2*cfg.inter, H, dtype=torch.bfloat16), torch.zeros(H, cfg.inter, dtype=torch.bfloat16), torch.zeros(H, dtype=torch.bfloat16))); self.mlp.to(DEV)
        self.L = []
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            hb = (p+"self_attn.q_proj.bias") in sd
            wqkv = np.concatenate([g(p+"self_attn.q_proj.weight"), g(p+"self_attn.k_proj.weight"), g(p+"self_attn.v_proj.weight")], 0)
            wgu = np.concatenate([g(p+"mlp.gate_proj.weight"), g(p+"mlp.up_proj.weight")], 0)
            bqkv = np.concatenate([g(p+"self_attn.q_proj.bias"), g(p+"self_attn.k_proj.bias"), g(p+"self_attn.v_proj.bias")]) if hb else None
            self.L.append(dict(in_norm=g(p+"input_layernorm.weight"), post_norm=g(p+"post_attention_layernorm.weight"),
                               wqkv=_dev(wqkv), bqkv=bqkv, wo=_dev(g(p+"self_attn.o_proj.weight")),
                               wgu=_dev(wgu), wd=_dev(g(p+"mlp.down_proj.weight"))))

    def _d(self, x): return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16).to(DEV)

    def layer(self, w, x, pos, ck, cv):
        c = self.c; hd = c.head_dim
        h = rmsnorm(x, w["in_norm"], c.eps)
        qkv = self.qkv_mm(w["wqkv"], self._d(h)).detach().float().cpu().numpy() + (w["bqkv"] if w["bqkv"] is not None else 0)
        q, k, v = qkv[:self.Hq], qkv[self.Hq:self.Hq+self.Hkv], qkv[self.Hq+self.Hkv:]
        q = rope(q, c.n_heads, pos, self.inv); k = rope(k, c.n_kv, pos, self.inv)
        ck.append(k); cv.append(v); K = np.stack(ck); V = np.stack(cv)
        attn = np.zeros(self.Hq, np.float32); s_ = math.sqrt(hd)
        for hh in range(c.n_heads):
            kv = hh // c.gqa
            s = (K[:, kv*hd:(kv+1)*hd] @ q[hh*hd:(hh+1)*hd]) / s_
            s -= s.max(); pp = np.exp(s); pp /= pp.sum()
            attn[hh*hd:(hh+1)*hd] = pp @ V[:, kv*hd:(kv+1)*hd]
        x = x + self.o_mm(w["wo"], self._d(attn)).detach().float().cpu().numpy()
        h2 = rmsnorm(x, w["post_norm"], c.eps)
        mlp = self.mlp(w["wgu"], w["wd"], self._d(h2)).detach().float().cpu().numpy()
        return x + mlp

    def argmax_per_pos(self, ids):
        ck = [[] for _ in range(self.c.n_layers)]; cv = [[] for _ in range(self.c.n_layers)]
        out = []
        for pos, t in enumerate(ids):
            x = self.embed[t].astype(np.float32).copy()
            for i in range(self.c.n_layers):
                x = self.layer(self.L[i], x, pos, ck[i], cv[i])
            out.append(int(np.argmax(self.lm @ rmsnorm(x, self.norm, self.c.eps))))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd); seq = ids + oracle.generate(ids, a.n); oam = oracle.argmax_per_pos(seq)
    print("[fast] fusion+mesh + resident + concat QKV + fused MLP block...", flush=True)
    t0 = time.time(); rt = FastRuntime(cfg, sd); load = time.time()-t0
    t0 = time.time(); am = rt.argmax_per_pos(seq); mt = (time.time()-t0)/len(seq)*1000
    lo = len(ids)-1; N = len(seq)-1-lo
    ok = sum(am[i] == oam[i] for i in range(lo, len(seq)-1))
    print("\n" + "=" * 56)
    print(f"{cfg.n_layers}L hidden={cfg.hidden} — 3 NPU launches/layer (QKV, O, MLP-block)")
    print(f"correctness: {ok}/{N} {'PASS ✅' if ok == N else 'CHECK'}")
    print(f"{mt:.0f} ms/token  (load+compile {load:.1f}s)")
    print("=" * 56)


if __name__ == "__main__":
    main()
