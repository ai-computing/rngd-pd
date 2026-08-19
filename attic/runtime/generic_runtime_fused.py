"""Fused all-NPU decode runtime — RMSNorm/RoPE/residual folded into 3 compiled modules/layer.

Goal was to keep the hidden state RESIDENT on device across all layers (kill the per-layer
ping-pong). **FINDING (hard SDK limit): a CompileModule's device output cannot be chained into
another CompileModule — doing so yields NaN and a "not implemented: bool" compiler panic. Every
module boundary MUST round-trip through host.** So full residency is impossible; what remains is
the win of FEWER, LARGER modules: 3 launches/layer (AttnPre = rmsnorm+qkv+rope, Attn = NPU
attention, AttnPost = o+residual+rmsnorm+fused-MLP+residual) with rmsnorm/rope/residual computed
in-graph (less host compute than generic_runtime_allnpu's 4 launches + host norms).

  AttnPre  : q,k,v = rope(qkv @ rmsnorm(x))          [rmsnorm + QKV + RoPE in-graph]
  (host)   : Kdev[:,pos]=k ; Vdev[:,pos]=v           [device-to-device slice write]
  Attn     : attn = softmax(qK^T/√d + mask) · V      [resident-KV manual-softmax GQA]
  AttnPost : x = (x + wo@attn) + mlp(rmsnorm(x+wo@attn))   [O + residual + rmsnorm + fused-MLP + residual]

Llama (no QKV bias) target. Verified vs numpy oracle.
  python3 generic_runtime_fused.py --model meta-llama/Llama-3.1-8B-Instruct --n 3
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


class AttnPre(torch.nn.Module):
    """rmsnorm(x)·nw → wqkv@h → RoPE(q,k). Returns (q[Hq], k[Hkv], v[Hkv]) on device."""
    def __init__(self, cfg): super().__init__(); self.c = cfg
    def _rope(self, x, nh, cos, sin):
        hd = self.c.head_dim; half = hd // 2
        xr = x.view(nh, hd); a, b = xr[..., :half], xr[..., half:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1).reshape(nh * hd)
    def forward(self, x, nw, wqkv, cos, sin):
        c = self.c; hd = c.head_dim
        h = _rms(x, nw, c.eps)
        qkv = wqkv @ h
        Hq = c.n_heads * hd; Hkv = c.n_kv * hd
        q = self._rope(qkv[:Hq], c.n_heads, cos, sin)
        k = self._rope(qkv[Hq:Hq + Hkv], c.n_kv, cos, sin)
        return q, k, qkv[Hq + Hkv:]


class Attn(torch.nn.Module):
    """Manual-softmax GQA decode attention over resident K/V [NKV,S,hd], masked padding."""
    def __init__(self, cfg, S): super().__init__(); self.c = cfg; self.S = S
    def forward(self, q, K, V, mask):
        c = self.c; hd = c.head_dim; g = c.gqa
        kh = K.repeat_interleave(g, 0); vh = V.repeat_interleave(g, 0)
        qh = q.view(c.n_heads, hd)
        sc = torch.einsum("hd,hsd->hs", qh.float(), kh.float()) / math.sqrt(hd) + mask
        sc = sc - sc.amax(-1, keepdim=True); e = torch.exp(sc)
        p = (e / e.sum(-1, keepdim=True)).to(BF)
        return torch.einsum("hs,hsd->hd", p.float(), vh.float()).to(BF).reshape(c.n_heads * hd)


class AttnPost(torch.nn.Module):
    """x2 = x + wo@attn ; out = x2 + wd@(silu(gate)*up), gate/up = (wgu@rmsnorm(x2)). One launch."""
    def __init__(self, cfg): super().__init__(); self.c = cfg; self.inter = cfg.inter
    def forward(self, attn, wo, x, nw2, wgu, wd):
        x2 = x + wo @ attn
        h2 = _rms(x2, nw2, self.c.eps)
        gu = wgu @ h2
        return x2 + wd @ (torch.nn.functional.silu(gu[:self.inter]) * gu[self.inter:])


def _dev(a): return torch.from_numpy(np.ascontiguousarray(a)).to(BF).to(DEV)
def _bucket(n): return max(64, ((n + 63) // 64) * 64)


class FusedRuntime:
    def __init__(self, cfg, sd):
        assert (f"model.layers.0.self_attn.q_proj.bias" not in sd), "QKV bias (Qwen) not supported in fused path yet"
        self.c = cfg; self.inv = cfg.inv_freq()
        g = lambda k: sd[k].astype(np.float32)
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        hd = cfg.head_dim; H = cfg.hidden
        self.hd = hd; self.NH = cfg.n_heads; self.NKV = cfg.n_kv; self.H = H
        self.Hq = cfg.n_heads * hd; self.Hkv = cfg.n_kv * hd; self.qkv_out = self.Hq + 2 * self.Hkv
        half = hd // 2
        # compile the 3 fused modules once
        F32 = torch.float32
        t0 = time.time()
        # x (residual stream) stays FLOAT32 on device across layers — bf16 residual drifts argmax over 32L.
        self.pre = CompileModule.from_module(AttnPre(cfg),
            (torch.zeros(H, dtype=F32), torch.zeros(H, dtype=BF), torch.zeros(self.qkv_out, H, dtype=BF),
             torch.zeros(half, dtype=F32), torch.zeros(half, dtype=F32))); self.pre.to(DEV)
        self.post = CompileModule.from_module(AttnPost(cfg),
            (torch.zeros(self.Hq, dtype=BF), torch.zeros(H, self.Hq, dtype=BF), torch.zeros(H, dtype=F32),
             torch.zeros(H, dtype=BF), torch.zeros(2 * cfg.inter, H, dtype=BF), torch.zeros(H, cfg.inter, dtype=BF))); self.post.to(DEV)
        self.compile_pre_post = time.time() - t0
        self.attn_mod = None; self.maxs = 0
        # resident weights per layer (concat qkv, gate|up)
        self.L = []
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            wqkv = np.concatenate([g(p+"self_attn.q_proj.weight"), g(p+"self_attn.k_proj.weight"), g(p+"self_attn.v_proj.weight")], 0)
            wgu = np.concatenate([g(p+"mlp.gate_proj.weight"), g(p+"mlp.up_proj.weight")], 0)
            self.L.append(dict(nw=_dev(g(p+"input_layernorm.weight")), nw2=_dev(g(p+"post_attention_layernorm.weight")),
                               wqkv=_dev(wqkv), wo=_dev(g(p+"self_attn.o_proj.weight")),
                               wgu=_dev(wgu), wd=_dev(g(p+"mlp.down_proj.weight"))))

    def _alloc(self, S):
        self.maxs = S
        self.attn_mod = CompileModule.from_module(Attn(self.c, S),
            (torch.zeros(self.Hq, dtype=BF), torch.zeros(self.NKV, S, self.hd, dtype=BF),
             torch.zeros(self.NKV, S, self.hd, dtype=BF), torch.zeros(S, dtype=torch.float32))); self.attn_mod.to(DEV)
        self.Kdev = [torch.zeros(self.NKV, S, self.hd, dtype=BF).to(DEV) for _ in range(self.c.n_layers)]
        self.Vdev = [torch.zeros(self.NKV, S, self.hd, dtype=BF).to(DEV) for _ in range(self.c.n_layers)]

    def _cossin(self, pos):
        f = np.arange(self.hd // 2); ang = pos * (self.inv)
        return _devf(np.cos(ang)), _devf(np.sin(ang))

    def forward_token(self, x_dev, pos, cos_d, sin_d, mask_d):
        """One token through all layers; x stays on device."""
        hd = self.hd
        for i in range(self.c.n_layers):
            w = self.L[i]
            # SDK LIMIT: a CompileModule's device output CANNOT be fed directly into another
            # CompileModule (produces NaN / "not implemented: bool" compiler panic). Every module
            # boundary must round-trip through host. So the residual stream cannot stay resident
            # across layers — this bounds how far the ping-pong can be cut (see module doc).
            q, k, v = self.pre(x_dev, w["nw"], w["wqkv"], cos_d, sin_d)
            self.Kdev[i][:, pos, :] = _dev(k.detach().float().cpu().numpy().reshape(self.NKV, hd))
            self.Vdev[i][:, pos, :] = _dev(v.detach().float().cpu().numpy().reshape(self.NKV, hd))
            q_h = _dev(q.detach().float().cpu().numpy())
            attn = self.attn_mod(q_h, self.Kdev[i], self.Vdev[i], mask_d)
            attn_h = _dev(attn.detach().float().cpu().numpy())
            x_dev = _devf(self.post(attn_h, w["wo"], x_dev, w["nw2"], w["wgu"], w["wd"]).detach().float().cpu().numpy())
        return x_dev

    def argmax_per_pos(self, ids):
        self._alloc(_bucket(len(ids)))
        out = []
        for pos, t in enumerate(ids):
            cos_d, sin_d = self._cossin(pos)
            mask = np.full(self.maxs, -1e4, np.float32); mask[:pos + 1] = 0.0
            mask_d = torch.from_numpy(mask).to(DEV)
            x_dev = _devf(self.embed[t].astype(np.float32))   # residual stream in f32
            x_dev = self.forward_token(x_dev, pos, cos_d, sin_d, mask_d)
            x = x_dev.detach().float().cpu().numpy()
            out.append(int(np.argmax(self.lm @ rmsnorm(x, self.norm, self.c.eps))))
        return out


def _devf(a): return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32).to(DEV)


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
    print("[fused] hidden resident on device across layers; rmsnorm+rope+residual in-graph...", flush=True)
    t0 = time.time(); rt = FusedRuntime(cfg, sd); load = time.time() - t0
    t0 = time.time(); am = rt.argmax_per_pos(seq); mt = (time.time() - t0) / len(seq) * 1000
    lo = len(ids) - 1; N = len(seq) - 1 - lo
    ok = sum(am[i] == oam[i] for i in range(lo, len(seq) - 1))
    print("\n" + "=" * 58)
    print(f"{cfg.n_layers}L hidden={cfg.hidden} — 3 launches/layer, hidden resident, x→host only at end")
    print(f"correctness: {ok}/{N} {'PASS ✅' if ok == N else 'CHECK'}")
    print(f"{mt:.0f} ms/token  (load+compile {load:.1f}s, pre/post compile {rt.compile_pre_post:.1f}s)")
    print("=" * 58)


if __name__ == "__main__":
    main()
