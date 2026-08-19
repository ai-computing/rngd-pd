"""Wired whole-DECODER-LAYER module — multi-level-optimization-guide.md principle #1.

Pack the whole layer into ONE nn.Module → ONE wired EDF → ONE launch/layer (no per-op host
round-trips). DECODE (1 query + KV cache) keeps the attention graph small enough to compile
(prefill S×S was what got killed before). KV update uses torch.cat (index_copy is NPU-unsupported):
the module returns (hidden, k_cur, v_cur) and the caller appends k/v to the host-side cache.

Module = rmsnorm→QKV→RoPE→[cat prior KV + current]→GQA manual-softmax attn→O+res→rmsnorm→
gate/up→SwiGLU→down+res. Weights+KV as inputs → compile ONCE, reuse all layers. fusion(8)+mesh(2).

  python3 generic_runtime_wired.py --model meta-llama/Llama-3.1-8B-Instruct --n 3
"""
import argparse, math, time, numpy as np, torch, sys
import _paths  # noqa: E402  (portable path bootstrap)
torch.set_grad_enabled(False)
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
from generic_runtime import ModelConfig, rmsnorm, GenericRuntime

DEV = "rngd:0"; BF = torch.bfloat16


def _rms(x, w, eps):
    v = (x.float() * x.float()).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + eps)).to(BF) * w


class DecoderLayer(torch.nn.Module):
    """One decode layer. Prior KV = [NKV,S,hd]; current token's k/v computed in-graph and cat'd."""
    def __init__(self, cfg, S): super().__init__(); self.c = cfg; self.S = S

    def _rope1(self, x, nh, cos, sin):
        hd = self.c.head_dim; half = hd // 2
        xr = x.view(nh, hd); a, b = xr[..., :half], xr[..., half:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1).reshape(nh * hd)

    def forward(self, x, n1, wqkv, wo, n2, wgu, wd, Kp, Vp, cos, sin, mask, onehot):
        c = self.c; hd = c.head_dim; g = c.gqa; Hq = c.n_heads * hd; Hkv = c.n_kv * hd; S = self.S
        h = _rms(x, n1, c.eps)
        qkv = wqkv @ h
        q = self._rope1(qkv[:Hq], c.n_heads, cos, sin)
        k = self._rope1(qkv[Hq:Hq + Hkv], c.n_kv, cos, sin)
        v = qkv[Hq + Hkv:]
        # Inject current k/v into slot `pos` via onehot broadcast-add (cat/index_copy → NPU fail)
        kc = Kp + onehot.view(1, S, 1) * k.view(c.n_kv, 1, hd)
        vc = Vp + onehot.view(1, S, 1) * v.view(c.n_kv, 1, hd)
        kh = kc.repeat_interleave(g, 0); vh = vc.repeat_interleave(g, 0)   # [NH,S,hd]
        qh = q.view(c.n_heads, hd)
        sc = torch.einsum("hd,hsd->hs", qh.float(), kh.float()) / math.sqrt(hd) + mask
        sc = sc - sc.amax(-1, keepdim=True); e = torch.exp(sc)
        p = (e / e.sum(-1, keepdim=True)).to(BF)
        attn = torch.einsum("hs,hsd->hd", p.float(), vh.float()).to(BF).reshape(Hq)
        x2 = x + (wo @ attn).float()
        h2 = _rms(x2, n2, c.eps)
        gu = wgu @ h2
        mlp = wd @ (torch.nn.functional.silu(gu[:c.inter]) * gu[c.inter:])
        return x2 + mlp.float(), k, v


def _dev(a): return torch.from_numpy(np.ascontiguousarray(a)).to(BF).to(DEV)
def _devf(a): return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32).to(DEV)
def _bucket(n): return max(64, ((n + 63) // 64) * 64)


class WiredRuntime:
    def __init__(self, cfg, sd, S):
        self.c = cfg; self.inv = cfg.inv_freq(); self.S = S
        g = lambda k: sd[k].astype(np.float32)
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        hd = cfg.head_dim; H = cfg.hidden
        self.hd = hd; self.NH = cfg.n_heads; self.NKV = cfg.n_kv
        self.Hq = cfg.n_heads * hd; self.Hkv = cfg.n_kv * hd; self.qkv_out = self.Hq + 2 * self.Hkv
        half = hd // 2
        z = lambda *s: torch.zeros(*s, dtype=BF)
        sample = (torch.zeros(H, dtype=torch.float32), z(H), z(self.qkv_out, H), z(H, self.Hq), z(H),
                  z(2 * cfg.inter, H), z(H, cfg.inter), z(self.NKV, S, hd), z(self.NKV, S, hd),
                  torch.zeros(half, dtype=torch.float32), torch.zeros(half, dtype=torch.float32),
                  torch.zeros(S, dtype=torch.float32), torch.zeros(S, dtype=torch.float32))
        t0 = time.time()
        self.m = CompileModule.from_module(DecoderLayer(cfg, S), sample); self.m.to(DEV)
        self.compile_s = time.time() - t0
        self.L = []
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            wqkv = np.concatenate([g(p+"self_attn.q_proj.weight"), g(p+"self_attn.k_proj.weight"), g(p+"self_attn.v_proj.weight")], 0)
            wgu = np.concatenate([g(p+"mlp.gate_proj.weight"), g(p+"mlp.up_proj.weight")], 0)
            self.L.append(dict(n1=_dev(g(p+"input_layernorm.weight")), n2=_dev(g(p+"post_attention_layernorm.weight")),
                               wqkv=_dev(wqkv), wo=_dev(g(p+"self_attn.o_proj.weight")),
                               wgu=_dev(wgu), wd=_dev(g(p+"mlp.down_proj.weight"))))

    def argmax_per_pos(self, ids):
        c = self.c; hd = self.hd; S = self.S
        ck = [np.zeros((self.NKV, S, hd), np.float32) for _ in range(c.n_layers)]
        cv = [np.zeros((self.NKV, S, hd), np.float32) for _ in range(c.n_layers)]
        out = []
        for pos, t in enumerate(ids):
            ang = pos * self.inv; cos = _devf(np.cos(ang)); sin = _devf(np.sin(ang))
            mask = np.full(S, -1e4, np.float32); mask[:pos + 1] = 0.0; mask_d = _devf(mask)   # valid 0..pos
            oh = np.zeros(S, np.float32); oh[pos] = 1.0; oh_d = _devf(oh)                       # inject at slot pos
            x = _devf(self.embed[t].astype(np.float32))
            for i in range(c.n_layers):
                w = self.L[i]
                hid, k, v = self.m(x, w["n1"], w["wqkv"], w["wo"], w["n2"], w["wgu"], w["wd"],
                                   _dev(ck[i]), _dev(cv[i]), cos, sin, mask_d, oh_d)
                x = hid
                # append current k/v to host cache at slot pos
                kn = k.detach().float().cpu().numpy().reshape(self.NKV, hd)
                vn = v.detach().float().cpu().numpy().reshape(self.NKV, hd)
                if pos < S: ck[i][:, pos, :] = kn; cv[i][:, pos, :] = vn
            out.append(int(np.argmax(self.lm @ rmsnorm(x.detach().float().cpu().numpy(), self.norm, c.eps))))
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
    S = _bucket(len(seq))
    print(f"[wired] compiling ONE whole-layer decode module (S={S}, fusion+mesh)...", flush=True)
    rt = WiredRuntime(cfg, sd, S)
    print(f"  compiled in {rt.compile_s:.1f}s", flush=True)
    t0 = time.time(); am = rt.argmax_per_pos(seq); mt = (time.time() - t0) / len(seq) * 1000
    lo = len(ids) - 1; N = len(seq) - 1 - lo
    ok = sum(am[i] == oam[i] for i in range(lo, len(seq) - 1))
    print("\n" + "=" * 58)
    print(f"{cfg.n_layers}L — 1 wired launch/layer (whole layer in ONE module, weights+KV reused)")
    print(f"correctness: {ok}/{N} {'PASS ✅' if ok == N else 'CHECK'}")
    print(f"{mt:.0f} ms/token  (compile {rt.compile_s:.1f}s)")
    print("=" * 58)


if __name__ == "__main__":
    main()
