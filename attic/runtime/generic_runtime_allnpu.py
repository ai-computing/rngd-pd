"""All-NPU decode runtime: matmuls AND attention on NPU (SDK CompileModule path).

Extends the FastRuntime stack (fusion+mesh + resident weights + concat QKV + fused MLP block)
by moving the previously-host attention onto the NPU via a bucketed manual-softmax
DecodeAttention CompileModule (masked padding for growing context). Only cheap O(H) element-
wise ops (RMSNorm, RoPE, residual) remain on host. The hand-VISA NPU softmax is broken; this
SDK compiled path (manual softmax IS fc.compile-able) sidesteps it and attacks the roadmap's
dominant bottleneck (host attention).

Verified vs numpy oracle (teacher-forced per-position argmax).
  python3 generic_runtime_allnpu.py --model meta-llama/Llama-3.1-8B-Instruct --n 3
"""
import argparse, math, time, numpy as np, torch, sys
import _paths  # noqa: E402  (portable path bootstrap)
torch.set_grad_enabled(False)
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
from generic_runtime import ModelConfig, rmsnorm, rope, GenericRuntime

DEV = "rngd:0"; BF = torch.bfloat16


class QKV(torch.nn.Module):
    def forward(self, w, x): return w @ x


class MLPBlock(torch.nn.Module):
    def __init__(self, inter): super().__init__(); self.inter = inter
    def forward(self, wgu, wd, x):
        gu = wgu @ x
        return wd @ (torch.nn.functional.silu(gu[:self.inter]) * gu[self.inter:])


class DecodeAttention(torch.nn.Module):
    """Manual-softmax GQA decode attention, fixed bucket S, masked padding."""
    def __init__(self, cfg, S): super().__init__(); self.c = cfg; self.S = S
    def forward(self, q, k_cache, v_cache, mask):
        c = self.c; hd = c.head_dim; g = c.gqa
        kh = k_cache.repeat_interleave(g, 0); vh = v_cache.repeat_interleave(g, 0)  # [NH,S,hd]
        sc = torch.einsum("hd,hsd->hs", q.float(), kh.float()) / math.sqrt(hd) + mask
        sc = sc - sc.amax(-1, keepdim=True); e = torch.exp(sc)
        p = (e / e.sum(-1, keepdim=True)).to(BF)
        return torch.einsum("hs,hsd->hd", p.float(), vh.float()).to(BF).reshape(c.n_heads * hd)


def _dev(a): return torch.from_numpy(np.ascontiguousarray(a)).to(BF).to(DEV)


def _bucket(n):  # smallest multiple of 64 >= n (min 64)
    return max(64, ((n + 63) // 64) * 64)


class AllNpuRuntime:
    def __init__(self, cfg, sd):
        self.c = cfg; self.inv = cfg.inv_freq()
        g = lambda k: sd[k].astype(np.float32)
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        hd = cfg.head_dim; H = cfg.hidden
        self.hd = hd; self.NH = cfg.n_heads; self.NKV = cfg.n_kv
        self.Hq = cfg.n_heads * hd; self.Hkv = cfg.n_kv * hd; self.qkv_out = self.Hq + 2 * self.Hkv
        self.qkv_mm = CompileModule.from_module(QKV(), (torch.zeros(self.qkv_out, H, dtype=BF), torch.zeros(H, dtype=BF))); self.qkv_mm.to(DEV)
        self.o_mm = CompileModule.from_module(QKV(), (torch.zeros(H, self.Hq, dtype=BF), torch.zeros(self.Hq, dtype=BF))); self.o_mm.to(DEV)
        self.mlp = CompileModule.from_module(MLPBlock(cfg.inter), (torch.zeros(2 * cfg.inter, H, dtype=BF), torch.zeros(H, cfg.inter, dtype=BF), torch.zeros(H, dtype=BF))); self.mlp.to(DEV)
        self.attn_mods = {}   # bucket S -> compiled DecodeAttention
        self.L = []
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."; hb = (p + "self_attn.q_proj.bias") in sd
            wqkv = np.concatenate([g(p+"self_attn.q_proj.weight"), g(p+"self_attn.k_proj.weight"), g(p+"self_attn.v_proj.weight")], 0)
            wgu = np.concatenate([g(p+"mlp.gate_proj.weight"), g(p+"mlp.up_proj.weight")], 0)
            bqkv = np.concatenate([g(p+"self_attn.q_proj.bias"), g(p+"self_attn.k_proj.bias"), g(p+"self_attn.v_proj.bias")]) if hb else None
            self.L.append(dict(in_norm=g(p+"input_layernorm.weight"), post_norm=g(p+"post_attention_layernorm.weight"),
                               wqkv=_dev(wqkv), bqkv=bqkv, wo=_dev(g(p+"self_attn.o_proj.weight")),
                               wgu=_dev(wgu), wd=_dev(g(p+"mlp.down_proj.weight"))))

    def _alloc_kv(self, S):
        """Resident per-layer device K/V buffers [NKV, S, hd] — filled by in-place row writes."""
        self.maxs = S
        self.attn_mod = CompileModule.from_module(DecodeAttention(self.c, S),
            (torch.zeros(self.NH, self.hd, dtype=BF), torch.zeros(self.NKV, S, self.hd, dtype=BF),
             torch.zeros(self.NKV, S, self.hd, dtype=BF), torch.zeros(S, dtype=torch.float32)))
        self.attn_mod.to(DEV)
        self.attn_mods = {S: self.attn_mod}
        self.Kdev = [torch.zeros(self.NKV, S, self.hd, dtype=BF).to(DEV) for _ in range(self.c.n_layers)]
        self.Vdev = [torch.zeros(self.NKV, S, self.hd, dtype=BF).to(DEV) for _ in range(self.c.n_layers)]

    def _d(self, x): return torch.from_numpy(np.ascontiguousarray(x)).to(BF).to(DEV)

    def layer(self, w, x, pos, li, mask_dev):
        c = self.c; hd = c.head_dim
        h = rmsnorm(x, w["in_norm"], c.eps)
        qkv = self.qkv_mm(w["wqkv"], self._d(h)).detach().float().cpu().numpy() + (w["bqkv"] if w["bqkv"] is not None else 0)
        q = qkv[:self.Hq]; k = qkv[self.Hq:self.Hq+self.Hkv]; v = qkv[self.Hq+self.Hkv:]
        q = rope(q, c.n_heads, pos, self.inv); k = rope(k, c.n_kv, pos, self.inv)
        # append ONLY the new k/v row into the resident device buffer (tiny upload, no re-upload of cache)
        self.Kdev[li][:, pos, :] = self._d(k.reshape(self.NKV, hd))
        self.Vdev[li][:, pos, :] = self._d(v.reshape(self.NKV, hd))
        attn = self.attn_mod(self._d(q.reshape(self.NH, hd)), self.Kdev[li], self.Vdev[li],
                             mask_dev).detach().float().cpu().numpy()
        x = x + self.o_mm(w["wo"], self._d(attn)).detach().float().cpu().numpy()
        h2 = rmsnorm(x, w["post_norm"], c.eps)
        mlp = self.mlp(w["wgu"], w["wd"], self._d(h2)).detach().float().cpu().numpy()
        return x + mlp

    def argmax_per_pos(self, ids):
        self._alloc_kv(_bucket(len(ids)))
        out = []
        for pos, t in enumerate(ids):
            mask = np.full(self.maxs, -1e4, np.float32); mask[:pos+1] = 0.0
            mask_dev = torch.from_numpy(mask).to(DEV)
            x = self.embed[t].astype(np.float32).copy()
            for i in range(self.c.n_layers):
                x = self.layer(self.L[i], x, pos, i, mask_dev)
            out.append(int(np.argmax(self.lm @ rmsnorm(x, self.norm, self.c.eps))))
        return out


def bench(model, ctxs):
    """Attention-only per-token cost across ALL layers: NPU (resident KV) vs host (numpy).
    Simulates one decode token: every layer does 1 attention over context C. This is the
    part that differs between FastRuntime (host) and AllNpuRuntime (NPU)."""
    from transformers import AutoConfig
    from npu_attention_probe import host_attn
    cfg = ModelConfig(AutoConfig.from_pretrained(model))
    hd = cfg.head_dim; NH = cfg.n_heads; NKV = cfg.n_kv; NL = cfg.n_layers
    d = lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).to(BF).to(DEV)
    print(f"{model}: {NL}L NH={NH} NKV={NKV} hd={hd}\n")
    print(f"{'ctx':>6} {'host(np) ms':>12} {'NPU ms':>9}  {'per-token attn (×'+str(NL)+'L)':>22}")
    rng = np.random.default_rng(0)
    for C in ctxs:
        S = _bucket(C)
        q = rng.standard_normal((NH, hd)).astype(np.float32) * 0.3
        kc = rng.standard_normal((NKV, C, hd)).astype(np.float32) * 0.3
        vc = rng.standard_normal((NKV, C, hd)).astype(np.float32) * 0.3
        # host: NL sequential numpy attentions (per-layer)
        for _ in range(2):
            for _ in range(NL): host_attn(cfg, q, kc, vc)
        t0 = time.time()
        for _ in range(5):
            for _ in range(NL): host_attn(cfg, q, kc, vc)
        ht = (time.time() - t0) / 5 * 1000
        # NPU: resident K/V per layer (allocate NL buffers of size S), NL attention launches
        m = CompileModule.from_module(DecodeAttention(cfg, S), (torch.zeros(NH, hd, dtype=BF), torch.zeros(NKV, S, hd, dtype=BF), torch.zeros(NKV, S, hd, dtype=BF), torch.zeros(S, dtype=torch.float32)))
        m.to(DEV)
        Kd = [torch.zeros(NKV, S, hd, dtype=BF).to(DEV) for _ in range(NL)]
        Vd = [torch.zeros(NKV, S, hd, dtype=BF).to(DEV) for _ in range(NL)]
        for kb in Kd: kb[:, :C, :] = d(kc)
        for vb in Vd: vb[:, :C, :] = d(vc)
        mask = np.full(S, -1e4, np.float32); mask[:C] = 0.0; md = torch.from_numpy(mask).to(DEV); qd = d(q)
        for _ in range(2):
            for li in range(NL): m(qd, Kd[li], Vd[li], md)
        t0 = time.time()
        for _ in range(5):
            for li in range(NL): _ = m(qd, Kd[li], Vd[li], md).detach().float().cpu().numpy()
        nt = (time.time() - t0) / 5 * 1000
        print(f"{C:>6} {ht:>12.2f} {nt:>9.2f}  {'NPU '+f'{ht/nt:.1f}x' if nt<ht else 'host '+f'{nt/ht:.1f}x':>22}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--bench-ctx", type=str, default="", help="comma ctx lengths: attention-only per-token cost across all layers, NPU vs host")
    a = ap.parse_args()
    if a.bench_ctx:
        return bench(a.model, [int(x) for x in a.bench_ctx.split(",")])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd); seq = ids + oracle.generate(ids, a.n); oam = oracle.argmax_per_pos(seq)
    print("[all-npu] fusion+mesh + resident + concat QKV + fused MLP + NPU attention...", flush=True)
    t0 = time.time(); rt = AllNpuRuntime(cfg, sd); load = time.time() - t0
    t0 = time.time(); am = rt.argmax_per_pos(seq); mt = (time.time() - t0) / len(seq) * 1000
    lo = len(ids) - 1; N = len(seq) - 1 - lo
    ok = sum(am[i] == oam[i] for i in range(lo, len(seq) - 1))
    print("\n" + "=" * 58)
    print(f"{cfg.n_layers}L hidden={cfg.hidden} — matmuls + ATTENTION on NPU (attn buckets {sorted(rt.attn_mods)})")
    print(f"correctness: {ok}/{N} {'PASS ✅' if ok == N else 'CHECK'}")
    print(f"{mt:.0f} ms/token  (load+compile {load:.1f}s)")
    print("=" * 58)


if __name__ == "__main__":
    main()
