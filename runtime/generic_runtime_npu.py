"""M2 — generic NPU runtime: ANY HF model (Llama/Qwen2 family) → fc.compile → run on NPU.

No hand-written per-dim kernels. Each matmul shape (out,in) is compiled ONCE via
furiosa.torch CompileModule with the weight as a runtime INPUT, then reused across all
layers. Weights resident on rngd:0 (bf16). RMS/RoPE/attention/SwiGLU on host (any arch).
KV cache owned by us (→ PD-disagg substrate). Verified vs the numpy oracle (generic_runtime).

  python generic_runtime_npu.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "..." --n 8
"""
import argparse, math, time, numpy as np, torch
import sys
import _paths  # noqa: E402  (portable path bootstrap)
from generic_runtime import ModelConfig, rmsnorm, rope, GenericRuntime
import furiosa.torch
from furiosa.torch.custom_ops import CompileModule

DEV = "rngd:0"


class _MatMul(torch.nn.Module):
    def forward(self, w, x): return w @ x


class MatMulCache:
    """One compiled EDF per (out,in) shape, reused across layers/models."""
    def __init__(self): self.cache = {}
    def get(self, out_d, in_d):
        key = (out_d, in_d)
        if key not in self.cache:
            m = CompileModule.from_module(
                _MatMul(), (torch.zeros(out_d, in_d, dtype=torch.bfloat16),
                            torch.zeros(in_d, dtype=torch.bfloat16)))
            m.to(DEV)
            self.cache[key] = m
        return self.cache[key]


def _dev(np_arr):
    return torch.from_numpy(np.ascontiguousarray(np_arr)).to(torch.bfloat16).to(DEV)


class GenericNpuRuntime:
    def __init__(self, cfg, sd):
        self.c = cfg; self.inv = cfg.inv_freq(); self.mm = MatMulCache()
        g = lambda k: sd[k].astype(np.float32)
        self.embed = g("model.embed_tokens.weight")
        self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        # pre-compile the handful of matmul shapes this model uses
        hd = cfg.head_dim
        for (o, i) in {(cfg.n_heads*hd, cfg.hidden), (cfg.n_kv*hd, cfg.hidden),
                       (cfg.hidden, cfg.n_heads*hd), (cfg.inter, cfg.hidden),
                       (cfg.hidden, cfg.inter)}:
            self.mm.get(o, i)
        # weights kept on HOST (bf16); streamed to the ~6GB furiosa.torch device pool
        # per matmul call (weight-as-input). Fits models bigger than the pool (e.g. 8B).
        hw = lambda k: torch.from_numpy(np.ascontiguousarray(g(k))).to(torch.bfloat16)
        self.L = []
        for idx in range(cfg.n_layers):
            p = f"model.layers.{idx}."
            hb = (p + "self_attn.q_proj.bias") in sd
            self.L.append(dict(
                in_norm=g(p+"input_layernorm.weight"), post_norm=g(p+"post_attention_layernorm.weight"),
                wq=hw(p+"self_attn.q_proj.weight"), wk=hw(p+"self_attn.k_proj.weight"),
                wv=hw(p+"self_attn.v_proj.weight"), wo=hw(p+"self_attn.o_proj.weight"),
                bq=g(p+"self_attn.q_proj.bias") if hb else None,
                bk=g(p+"self_attn.k_proj.bias") if hb else None,
                bv=g(p+"self_attn.v_proj.bias") if hb else None,
                wg=hw(p+"mlp.gate_proj.weight"), wu=hw(p+"mlp.up_proj.weight"),
                wd=hw(p+"mlp.down_proj.weight")))

    def _mm(self, w_host, x_np, out_d, in_d):
        m = self.mm.get(out_d, in_d)
        w = w_host.to(DEV)                                   # stream weight (transient)
        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(torch.bfloat16).to(DEV)
        return m(w, x).float().cpu().numpy()

    def layer(self, w, x, pos, ck, cv):
        c = self.c; hd = c.head_dim; H = c.hidden
        h = rmsnorm(x, w["in_norm"], c.eps)
        q = self._mm(w["wq"], h, c.n_heads*hd, H) + (w["bq"] if w["bq"] is not None else 0)
        k = self._mm(w["wk"], h, c.n_kv*hd, H) + (w["bk"] if w["bk"] is not None else 0)
        v = self._mm(w["wv"], h, c.n_kv*hd, H) + (w["bv"] if w["bv"] is not None else 0)
        q = rope(q, c.n_heads, pos, self.inv); k = rope(k, c.n_kv, pos, self.inv)
        ck.append(k); cv.append(v)
        K = np.stack(ck); V = np.stack(cv); attn = np.zeros(c.n_heads*hd, np.float32); sd_ = math.sqrt(hd)
        for hh in range(c.n_heads):
            kv = hh // c.gqa
            s = (K[:, kv*hd:(kv+1)*hd] @ q[hh*hd:(hh+1)*hd]) / sd_
            s -= s.max(); p = np.exp(s); p /= p.sum()
            attn[hh*hd:(hh+1)*hd] = p @ V[:, kv*hd:(kv+1)*hd]
        x = x + self._mm(w["wo"], attn, H, c.n_heads*hd)
        h2 = rmsnorm(x, w["post_norm"], c.eps)
        g = self._mm(w["wg"], h2, c.inter, H); u = self._mm(w["wu"], h2, c.inter, H)
        return x + self._mm(w["wd"], (g/(1.0+np.exp(-g)))*u, H, c.inter)

    def _fwd(self, tok, pos, ck, cv):
        x = self.embed[tok].astype(np.float32).copy()
        for i in range(self.c.n_layers):
            x = self.layer(self.L[i], x, pos, ck[i], cv[i])
        return x

    def argmax_per_pos(self, ids):
        ck = [[] for _ in range(self.c.n_layers)]; cv = [[] for _ in range(self.c.n_layers)]
        out = []
        for pos, t in enumerate(ids):
            x = self._fwd(t, pos, ck, cv)
            out.append(int(np.argmax(self.lm @ rmsnorm(x, self.norm, self.c.eps))))
        return out

    # ---- serving API (used by generic_server.py) ----
    def _logit_argmax(self, x):
        return int(np.argmax(self.lm @ rmsnorm(x, self.norm, self.c.eps)))

    def prefill(self, ids):
        """Process prompt, build KV cache we own. Returns (ck, cv, first_token, next_pos)."""
        ck = [[] for _ in range(self.c.n_layers)]; cv = [[] for _ in range(self.c.n_layers)]
        x = None
        for pos, t in enumerate(ids):
            x = self._fwd(t, pos, ck, cv)
        return ck, cv, self._logit_argmax(x), len(ids)

    def decode(self, ck, cv, tok, pos, n_new):
        """Continue generation from an existing KV cache. Returns generated token ids."""
        gen = [tok]
        for _ in range(n_new - 1):
            x = self._fwd(tok, pos, ck, cv)
            tok = self._logit_argmax(x); gen.append(tok); pos += 1
        return gen

    def generate(self, ids, n_new):
        ck, cv, first, pos = self.prefill(ids)
        return self.decode(ck, cv, first, pos, n_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf
    ids = tk.encode(a.prompt, add_special_tokens=False)

    print(f"[oracle] numpy (ground truth)...", flush=True)
    oracle = GenericRuntime(cfg, sd)
    # teacher-force over oracle's own greedy sequence
    seq = ids + oracle.generate(ids, a.n)
    oracle_am = oracle.argmax_per_pos(seq)

    print(f"[npu] compiling matmul shapes + loading weights to rngd:0...", flush=True)
    t0 = time.time(); rt = GenericNpuRuntime(cfg, sd); load = time.time() - t0
    t0 = time.time(); npu_am = rt.argmax_per_pos(seq); mt = (time.time()-t0)/len(seq)

    lo = len(ids) - 1
    agree = sum(npu_am[i] == oracle_am[i] for i in range(lo, len(seq)-1)); tot = len(seq)-1-lo
    print("\n" + "=" * 58)
    print(f"model: {cfg.n_layers}L hidden={cfg.hidden} GQA {cfg.n_heads}/{cfg.n_kv} — {len(rt.mm.cache)} matmul shapes compiled")
    print(f"NPU vs numpy-oracle teacher-forced argmax: {agree}/{tot} "
          f"{'PASS ✅' if agree==tot else 'CHECK'}")
    print(f"NPU: {load:.1f}s load+compile, {mt*1000:.0f} ms/token")
    print("text:", repr(tk.decode(seq)))
    print("=" * 58)


if __name__ == "__main__":
    main()
