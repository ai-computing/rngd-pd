"""M6 — perf optimizations on the generic NPU runtime (dim-general, verified).

Two model-agnostic wins over generic_runtime_npu.py (which streamed weights + 7 matmuls/layer):
  (1) RESIDENT weights: keep weights on the device pool (models that fit ~6GB) → no
      per-matmul re-upload.
  (2) WEIGHT CONCAT: fuse q|k|v → one matmul, gate|up → one matmul  (7 → 4 launches/layer).
Both are just host-side concatenation + resident upload — no new kernels, any Llama/Qwen2 dim.

Benchmarks streaming-7 (baseline) vs resident-concat-4, verified vs the numpy oracle.
  python generic_runtime_opt.py --model Qwen/Qwen2.5-0.5B-Instruct --n 8
"""
import argparse, math, time, numpy as np, torch, sys
import _paths  # noqa: E402  (portable path bootstrap)
from generic_runtime import ModelConfig, rmsnorm, rope, GenericRuntime
import generic_runtime_npu as GN
import furiosa.torch
from furiosa.torch.custom_ops import CompileModule

DEV = "rngd:0"


class _MatMul(torch.nn.Module):
    def forward(self, w, x): return w @ x


class OptRuntime:
    """Resident weights + concatenated QKV / gate-up matmuls."""
    def __init__(self, cfg, sd):
        self.c = cfg; self.inv = cfg.inv_freq(); self.mm = {}
        g = lambda k: sd[k].astype(np.float32)
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        hd = cfg.head_dim; H = cfg.hidden
        self.qkv_out = cfg.n_heads*hd + 2*cfg.n_kv*hd
        self.gu_out = 2*cfg.inter
        for (o, i) in {(self.qkv_out, H), (H, cfg.n_heads*hd), (self.gu_out, H), (H, cfg.inter)}:
            self._compile(o, i)
        dev = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16).to(DEV)
        self.L = []
        for idx in range(cfg.n_layers):
            p = f"model.layers.{idx}."
            hb = (p+"self_attn.q_proj.bias") in sd
            wqkv = np.concatenate([g(p+"self_attn.q_proj.weight"), g(p+"self_attn.k_proj.weight"),
                                   g(p+"self_attn.v_proj.weight")], axis=0)   # [qkv_out, H]
            wgu = np.concatenate([g(p+"mlp.gate_proj.weight"), g(p+"mlp.up_proj.weight")], axis=0)
            bqkv = (np.concatenate([g(p+"self_attn.q_proj.bias"), g(p+"self_attn.k_proj.bias"),
                                    g(p+"self_attn.v_proj.bias")]) if hb else None)
            self.L.append(dict(
                in_norm=g(p+"input_layernorm.weight"), post_norm=g(p+"post_attention_layernorm.weight"),
                wqkv=dev(wqkv), bqkv=bqkv, wo=dev(g(p+"self_attn.o_proj.weight")),
                wgu=dev(wgu), wd=dev(g(p+"mlp.down_proj.weight"))))

    def _compile(self, o, i):
        if (o, i) not in self.mm:
            m = CompileModule.from_module(_MatMul(), (torch.zeros(o, i, dtype=torch.bfloat16),
                                                      torch.zeros(i, dtype=torch.bfloat16)))
            m.to(DEV); self.mm[(o, i)] = m

    def _mm(self, w_dev, x_np, o, i):
        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(torch.bfloat16).to(DEV)
        return self.mm[(o, i)](w_dev, x).float().cpu().numpy()

    def layer(self, w, x, pos, ck, cv):
        c = self.c; hd = c.head_dim; H = c.hidden; Hq = c.n_heads*hd; Hkv = c.n_kv*hd
        h = rmsnorm(x, w["in_norm"], c.eps)
        qkv = self._mm(w["wqkv"], h, self.qkv_out, H) + (w["bqkv"] if w["bqkv"] is not None else 0)
        q, k, v = qkv[:Hq], qkv[Hq:Hq+Hkv], qkv[Hq+Hkv:]
        q = rope(q, c.n_heads, pos, self.inv); k = rope(k, c.n_kv, pos, self.inv)
        ck.append(k); cv.append(v); K = np.stack(ck); V = np.stack(cv)
        attn = np.zeros(Hq, np.float32); s_ = math.sqrt(hd)
        for hh in range(c.n_heads):
            kv = hh // c.gqa
            s = (K[:, kv*hd:(kv+1)*hd] @ q[hh*hd:(hh+1)*hd]) / s_
            s -= s.max(); pp = np.exp(s); pp /= pp.sum()
            attn[hh*hd:(hh+1)*hd] = pp @ V[:, kv*hd:(kv+1)*hd]
        x = x + self._mm(w["wo"], attn, H, Hq)
        h2 = rmsnorm(x, w["post_norm"], c.eps)
        gu = self._mm(w["wgu"], h2, self.gu_out, H)
        gate, up = gu[:c.inter], gu[c.inter:]
        return x + self._mm(w["wd"], (gate/(1.0+np.exp(-gate)))*up, H, c.inter)

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
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd)
    seq = ids + oracle.generate(ids, a.n); oam = oracle.argmax_per_pos(seq)

    # baseline: streaming, 7 matmuls/layer
    GN.DEV = DEV
    base = GN.GenericNpuRuntime(cfg, sd)
    t = time.time(); bam = base.argmax_per_pos(seq); bt = (time.time()-t)/len(seq)*1000
    # opt: resident + concat, 4 matmuls/layer
    opt = OptRuntime(cfg, sd)
    t = time.time(); oam2 = opt.argmax_per_pos(seq); ot = (time.time()-t)/len(seq)*1000

    lo = len(ids)-1; N = len(seq)-1-lo
    bok = sum(bam[i] == oam[i] for i in range(lo, len(seq)-1))
    ook = sum(oam2[i] == oam[i] for i in range(lo, len(seq)-1))
    print("\n" + "=" * 56)
    print(f"baseline (stream, 7 mm/layer): {bt:.0f} ms/tok  correctness {bok}/{N}")
    print(f"opt (resident, concat 4 mm/layer): {ot:.0f} ms/tok  correctness {ook}/{N}")
    print(f"speedup: {bt/ot:.2f}×  (dim-general: weight concat + resident, no new kernels)")
    print("=" * 56)


if __name__ == "__main__":
    main()
