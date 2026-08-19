"""Adaptive attention runtime — host attention for short context, NPU attention for long.

Deployable culmination. NPU decode attention (resident KV, manual-softmax CompileModule) is flat
~fast regardless of context; host numpy attention is cheaper at short context but blows up O(S).
Crossover ≈ 512-1024 (see generic_runtime_allnpu.py --bench-ctx). This runtime keeps BOTH a host
K/V copy and a resident device K/V buffer (append one small row per token to each), and per layer
picks the cheaper attention path by current context length. Short ctx → FastRuntime-class latency;
long ctx (RAG/long-doc) → NPU attention avoids the host blow-up.

Matmuls + MLP always on NPU (fusion+mesh+resident+concat+fused-MLP, from FastRuntime). Verified
vs numpy oracle.
  python3 generic_runtime_adaptive.py --model meta-llama/Llama-3.1-8B-Instruct --n 3 --crossover 512
"""
import argparse, math, time, numpy as np, torch, sys
import _paths  # noqa: E402  (portable path bootstrap)
torch.set_grad_enabled(False)
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
from generic_runtime import ModelConfig, rmsnorm, rope, GenericRuntime
from generic_runtime_allnpu import QKV, MLPBlock, DecodeAttention, _bucket

DEV = "rngd:0"; BF = torch.bfloat16


def _dev(a): return torch.from_numpy(np.ascontiguousarray(a)).to(BF).to(DEV)


class AdaptiveRuntime:
    def __init__(self, cfg, sd, crossover=512):
        self.c = cfg; self.inv = cfg.inv_freq(); self.crossover = crossover
        g = lambda k: sd[k].astype(np.float32)
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        hd = cfg.head_dim; H = cfg.hidden
        self.hd = hd; self.NH = cfg.n_heads; self.NKV = cfg.n_kv
        self.Hq = cfg.n_heads * hd; self.Hkv = cfg.n_kv * hd; self.qkv_out = self.Hq + 2 * self.Hkv
        self.qkv_mm = CompileModule.from_module(QKV(), (torch.zeros(self.qkv_out, H, dtype=BF), torch.zeros(H, dtype=BF))); self.qkv_mm.to(DEV)
        self.o_mm = CompileModule.from_module(QKV(), (torch.zeros(H, self.Hq, dtype=BF), torch.zeros(self.Hq, dtype=BF))); self.o_mm.to(DEV)
        self.mlp = CompileModule.from_module(MLPBlock(cfg.inter), (torch.zeros(2 * cfg.inter, H, dtype=BF), torch.zeros(H, cfg.inter, dtype=BF), torch.zeros(H, dtype=BF))); self.mlp.to(DEV)
        self.attn_mod = None; self.maxs = 0
        self.n_host = 0; self.n_npu = 0
        self.L = []
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."; hb = (p + "self_attn.q_proj.bias") in sd
            wqkv = np.concatenate([g(p+"self_attn.q_proj.weight"), g(p+"self_attn.k_proj.weight"), g(p+"self_attn.v_proj.weight")], 0)
            wgu = np.concatenate([g(p+"mlp.gate_proj.weight"), g(p+"mlp.up_proj.weight")], 0)
            bqkv = np.concatenate([g(p+"self_attn.q_proj.bias"), g(p+"self_attn.k_proj.bias"), g(p+"self_attn.v_proj.bias")]) if hb else None
            self.L.append(dict(in_norm=g(p+"input_layernorm.weight"), post_norm=g(p+"post_attention_layernorm.weight"),
                               wqkv=_dev(wqkv), bqkv=bqkv, wo=_dev(g(p+"self_attn.o_proj.weight")),
                               wgu=_dev(wgu), wd=_dev(g(p+"mlp.down_proj.weight"))))

    def _alloc(self, S):
        self.maxs = S
        self.attn_mod = CompileModule.from_module(DecodeAttention(self.c, S),
            (torch.zeros(self.NH, self.hd, dtype=BF), torch.zeros(self.NKV, S, self.hd, dtype=BF),
             torch.zeros(self.NKV, S, self.hd, dtype=BF), torch.zeros(S, dtype=torch.float32))); self.attn_mod.to(DEV)
        self.Kdev = [torch.zeros(self.NKV, S, self.hd, dtype=BF).to(DEV) for _ in range(self.c.n_layers)]
        self.Vdev = [torch.zeros(self.NKV, S, self.hd, dtype=BF).to(DEV) for _ in range(self.c.n_layers)]

    def _d(self, x): return torch.from_numpy(np.ascontiguousarray(x)).to(BF).to(DEV)

    def layer(self, w, x, pos, li, ckh, cvh, mask_d, use_npu):
        c = self.c; hd = c.head_dim
        h = rmsnorm(x, w["in_norm"], c.eps)
        qkv = self.qkv_mm(w["wqkv"], self._d(h)).detach().float().cpu().numpy() + (w["bqkv"] if w["bqkv"] is not None else 0)
        q = qkv[:self.Hq]; k = qkv[self.Hq:self.Hq+self.Hkv]; v = qkv[self.Hq+self.Hkv:]
        q = rope(q, c.n_heads, pos, self.inv); k = rope(k, c.n_kv, pos, self.inv)
        ckh.append(k); cvh.append(v)                              # host K/V (always; cheap, enables backfill)
        if use_npu:                                               # NPU attention (flat, wins long ctx)
            self.Kdev[li][:, pos, :] = self._d(k.reshape(self.NKV, hd))   # append 1 row to resident device KV
            self.Vdev[li][:, pos, :] = self._d(v.reshape(self.NKV, hd))
            attn = self.attn_mod(self._d(q.reshape(self.NH, hd)), self.Kdev[li], self.Vdev[li],
                                 mask_d).detach().float().cpu().numpy()
        else:                                                     # HOST attention (cheap for short ctx)
            K = np.stack(ckh); V = np.stack(cvh); attn = np.zeros(self.Hq, np.float32); s_ = math.sqrt(hd)
            for hh in range(c.n_heads):
                kv = hh // c.gqa
                s = (K[:, kv*hd:(kv+1)*hd] @ q[hh*hd:(hh+1)*hd]) / s_
                s -= s.max(); pp = np.exp(s); pp /= pp.sum()
                attn[hh*hd:(hh+1)*hd] = pp @ V[:, kv*hd:(kv+1)*hd]
        x = x + self.o_mm(w["wo"], self._d(attn)).detach().float().cpu().numpy()
        h2 = rmsnorm(x, w["post_norm"], c.eps)
        mlp = self.mlp(w["wgu"], w["wd"], self._d(h2)).detach().float().cpu().numpy()
        return x + mlp

    def _backfill(self, li, ckh, cvh, upto):
        """One-time bulk upload of host K/V rows 0..upto-1 into the resident device buffer."""
        K = np.stack(ckh).reshape(upto, self.NKV, self.hd).transpose(1, 0, 2)
        V = np.stack(cvh).reshape(upto, self.NKV, self.hd).transpose(1, 0, 2)
        self.Kdev[li][:, :upto, :] = self._d(K); self.Vdev[li][:, :upto, :] = self._d(V)

    def argmax_per_pos(self, ids):
        self._alloc(_bucket(len(ids)))
        ckh = [[] for _ in range(self.c.n_layers)]; cvh = [[] for _ in range(self.c.n_layers)]
        out = []; on_npu = False
        for pos, t in enumerate(ids):
            use_npu = (pos + 1) > self.crossover
            if use_npu and not on_npu:                            # transition: backfill device KV once
                for i in range(self.c.n_layers): self._backfill(i, ckh[i], cvh[i], pos)
                on_npu = True
            if use_npu: self.n_npu += 1
            else: self.n_host += 1
            mask_d = torch.from_numpy(np.where(np.arange(self.maxs) < pos + 1, 0.0, -1e4).astype(np.float32)).to(DEV) if use_npu else None
            x = self.embed[t].astype(np.float32).copy()
            for i in range(self.c.n_layers):
                x = self.layer(self.L[i], x, pos, i, ckh[i], cvh[i], mask_d, use_npu)
            out.append(int(np.argmax(self.lm @ rmsnorm(x, self.norm, self.c.eps))))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--crossover", type=int, default=512)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd); seq = ids + oracle.generate(ids, a.n); oam = oracle.argmax_per_pos(seq)
    print(f"[adaptive] host attn if ctx<={a.crossover} else NPU attn (matmuls always NPU)...", flush=True)
    t0 = time.time(); rt = AdaptiveRuntime(cfg, sd, a.crossover); load = time.time() - t0
    t0 = time.time(); am = rt.argmax_per_pos(seq); mt = (time.time() - t0) / len(seq) * 1000
    lo = len(ids) - 1; N = len(seq) - 1 - lo
    ok = sum(am[i] == oam[i] for i in range(lo, len(seq) - 1))
    print("\n" + "=" * 58)
    print(f"{cfg.n_layers}L hidden={cfg.hidden} — adaptive attention (crossover ctx={a.crossover})")
    print(f"tokens via host-attn: {rt.n_host}, via NPU-attn: {rt.n_npu}")
    print(f"correctness: {ok}/{N} {'PASS ✅' if ok == N else 'CHECK'}")
    print(f"{mt:.0f} ms/token  (load+compile {load:.1f}s)")
    print("=" * 58)


if __name__ == "__main__":
    main()
