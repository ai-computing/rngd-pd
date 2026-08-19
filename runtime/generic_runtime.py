"""M1 — config-driven generic decoder runtime (Llama/Qwen2 family), verified vs HF.

Same code, any {Llama3, Qwen2.5, Mistral, EXAONE-4}-style model: dims + variations
(QKV bias, tied embeddings, RoPE theta, llama3 rope scaling, GQA ratio, head_dim) all
come from the HF config + state_dict. Numpy oracle (CPU) — the correctness foundation
that the NPU runtime (qwen_runtime_npu.py) mirrors. Verified token-for-token vs HF.

  python generic_runtime.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "The capital of France is" --n 12
  python generic_runtime.py --model meta-llama/Llama-3.1-8B-Instruct --prompt "..." --n 4
"""
import argparse, math, time
import numpy as np


class ModelConfig:
    def __init__(self, hf):
        self.n_layers = hf.num_hidden_layers
        self.hidden = hf.hidden_size
        self.n_heads = hf.num_attention_heads
        self.n_kv = getattr(hf, "num_key_value_heads", self.n_heads)
        self.head_dim = getattr(hf, "head_dim", None) or self.hidden // self.n_heads
        self.inter = hf.intermediate_size
        self.eps = getattr(hf, "rms_norm_eps", 1e-6)
        self.vocab = hf.vocab_size
        self.tie = getattr(hf, "tie_word_embeddings", False)
        # RoPE: theta may live under rope_scaling (llama3)
        rs = getattr(hf, "rope_scaling", None) or {}
        self.theta = getattr(hf, "rope_theta", None) or rs.get("rope_theta") or 10000.0
        self.rope_scaling = rs if rs.get("rope_type") == "llama3" else None
        self.gqa = self.n_heads // self.n_kv

    def inv_freq(self):
        d = self.head_dim
        inv = 1.0 / (self.theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        rs = self.rope_scaling
        if rs:                                      # llama3 frequency scaling
            factor, lo, hi, orig = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], rs["original_max_position_embeddings"]
            low_wl, high_wl = orig / lo, orig / hi
            wl = 2 * math.pi / inv
            out = inv.copy()
            for i, w in enumerate(wl):
                if w > low_wl:
                    out[i] = inv[i] / factor
                elif w < high_wl:
                    out[i] = inv[i]
                else:
                    s = (orig / w - lo) / (hi - lo)
                    out[i] = (1 - s) * inv[i] / factor + s * inv[i]
            inv = out
        return inv.astype(np.float32)


def rmsnorm(x, w, eps):
    return (x / math.sqrt(float(np.mean(x * x)) + eps)) * w


def rope(x, n_heads, pos, inv_freq):
    hd = x.shape[0] // n_heads
    half = hd // 2
    ang = pos * inv_freq                            # [half]
    cos, sin = np.cos(ang), np.sin(ang)
    xr = x.reshape(n_heads, hd).copy()
    a, b = xr[:, :half].copy(), xr[:, half:].copy()  # rotate_half (HF)
    xr[:, :half] = a * cos - b * sin
    xr[:, half:] = b * cos + a * sin
    return xr.reshape(-1)


class GenericRuntime:
    def __init__(self, cfg, sd, dtype=np.float32):
        self.c = cfg; self.inv = cfg.inv_freq()
        g = lambda k: sd[k].astype(dtype)
        self.embed = g("model.embed_tokens.weight")
        self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        self.L = []
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            has_bias = (p + "self_attn.q_proj.bias") in sd
            self.L.append(dict(
                in_norm=g(p + "input_layernorm.weight"),
                post_norm=g(p + "post_attention_layernorm.weight"),
                wq=g(p + "self_attn.q_proj.weight"), wk=g(p + "self_attn.k_proj.weight"),
                wv=g(p + "self_attn.v_proj.weight"), wo=g(p + "self_attn.o_proj.weight"),
                bq=g(p + "self_attn.q_proj.bias") if has_bias else None,
                bk=g(p + "self_attn.k_proj.bias") if has_bias else None,
                bv=g(p + "self_attn.v_proj.bias") if has_bias else None,
                wg=g(p + "mlp.gate_proj.weight"), wu=g(p + "mlp.up_proj.weight"),
                wd=g(p + "mlp.down_proj.weight")))

    def layer(self, w, x, pos, ck, cv):
        c = self.c; hd = c.head_dim
        h = rmsnorm(x, w["in_norm"], c.eps)
        q = w["wq"] @ h + (w["bq"] if w["bq"] is not None else 0)
        k = w["wk"] @ h + (w["bk"] if w["bk"] is not None else 0)
        v = w["wv"] @ h + (w["bv"] if w["bv"] is not None else 0)
        q = rope(q, c.n_heads, pos, self.inv); k = rope(k, c.n_kv, pos, self.inv)
        ck.append(k); cv.append(v)
        K = np.stack(ck); V = np.stack(cv)
        attn = np.zeros(c.n_heads * hd, np.float32); sd_ = math.sqrt(hd)
        for hh in range(c.n_heads):
            kv = hh // c.gqa
            qh = q[hh*hd:(hh+1)*hd]
            kh = K[:, kv*hd:(kv+1)*hd]; vh = V[:, kv*hd:(kv+1)*hd]
            s = (kh @ qh) / sd_; s -= s.max(); p = np.exp(s); p /= p.sum()
            attn[hh*hd:(hh+1)*hd] = p @ vh
        x = x + w["wo"] @ attn
        h2 = rmsnorm(x, w["post_norm"], c.eps)
        g = w["wg"] @ h2; u = w["wu"] @ h2
        return x + w["wd"] @ ((g / (1.0 + np.exp(-g))) * u)

    def generate(self, ids, n_new):
        ck = [[] for _ in range(self.c.n_layers)]; cv = [[] for _ in range(self.c.n_layers)]
        x = None
        for pos, t in enumerate(ids):
            x = self._fwd(t, pos, ck, cv)
        gen = []; tok, pos = int(np.argmax(self.lm @ rmsnorm(x, self.norm, self.c.eps))), len(ids)
        gen.append(tok)
        for _ in range(n_new - 1):
            x = self._fwd(tok, pos, ck, cv)
            tok = int(np.argmax(self.lm @ rmsnorm(x, self.norm, self.c.eps))); gen.append(tok); pos += 1
        return gen

    def _fwd(self, tok, pos, ck, cv):
        x = self.embed[tok].astype(np.float32).copy()
        for i in range(self.c.n_layers):
            x = self.layer(self.L[i], x, pos, ck[i], cv[i])
        return x

    def argmax_per_pos(self, ids):
        """Teacher-forced: argmax at each position given the true prefix (decouples
        from greedy divergence — the right correctness metric vs a reference)."""
        ck = [[] for _ in range(self.c.n_layers)]; cv = [[] for _ in range(self.c.n_layers)]
        out = []
        for pos, t in enumerate(ids):
            x = self._fwd(t, pos, ck, cv)
            out.append(int(np.argmax(self.lm @ rmsnorm(x, self.norm, self.c.eps))))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    print(f"loading {a.model} (CPU)...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    hf.eval()
    cfg = ModelConfig(hf.config)
    print(f"config: {cfg.n_layers}L hidden={cfg.hidden} heads={cfg.n_heads}/{cfg.n_kv} "
          f"hd={cfg.head_dim} inter={cfg.inter} theta={cfg.theta:g} "
          f"llama3_rope={'Y' if cfg.rope_scaling else 'N'} bias={'Y' if 'model.layers.0.self_attn.q_proj.bias' in hf.state_dict() else 'N'} tied={cfg.tie}", flush=True)

    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    rt = GenericRuntime(cfg, sd)
    ids = tk.encode(a.prompt, add_special_tokens=False)

    # HF greedy reference sequence
    with torch.no_grad():
        hf_gen = hf.generate(torch.tensor([ids]), max_new_tokens=a.n, do_sample=False,
                             use_cache=True)[0, len(ids):].tolist()
    seq = ids + hf_gen

    # (1) free-running greedy (expected to diverge on near-ties — numeric chaos)
    t0 = time.time(); mine = rt.generate(ids, a.n); mt = time.time() - t0
    greedy_match = sum(x == y for x, y in zip(mine, hf_gen))

    # (2) TEACHER-FORCED per-position argmax agreement vs HF (real correctness metric)
    with torch.no_grad():
        hf_logits = hf(torch.tensor([seq])).logits[0]          # [S, vocab]
        hf_am = hf_logits.argmax(-1).tolist()
    my_am = rt.argmax_per_pos(seq)
    lo = len(ids) - 1
    agree = sum(my_am[i] == hf_am[i] for i in range(lo, len(seq) - 1))
    tot = len(seq) - 1 - lo

    print("\n" + "=" * 60)
    print("free-greedy generic:", mine)
    print("free-greedy HF     :", hf_gen)
    print(f"  greedy match {greedy_match}/{len(mine)} (diverges on near-ties = numeric)")
    print(f"TEACHER-FORCED argmax agreement vs HF: {agree}/{tot} "
          f"({'PASS ✅' if agree == tot else 'CHECK'})  ← correctness metric")
    print(f"({mt/a.n*1000:.0f} ms/tok numpy)")
    print("text:", repr(tk.decode(seq)))
    print("=" * 60)


if __name__ == "__main__":
    main()
