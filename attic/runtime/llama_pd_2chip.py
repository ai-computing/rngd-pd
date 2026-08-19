"""REAL cross-chip PD disaggregation — prefill@chip0 PRODUCES KV → moves to chip1 → decode@chip1.

Unlike the earlier "concurrent" test (which used INDEPENDENT random KV per chip — no handoff), here
the KV genuinely MOVES: prefill on chip0 computes the prompt's K/V (the wired DecoderLayer OUTPUTS
k/v), those tensors cross chip0→host→chip1, and decode on chip1 continues from THAT KV. Verified ==
monolithic; measures prefill time, KV size, chip0→chip1 transfer, decode time.

Uses generic_runtime_wired.DecoderLayer (raw-torch wired layer, outputs (hidden,k,v), 3/3 correct).
Two CompileModules — one .to(rngd:0), one .to(rngd:1). Weights replicated per chip.
"""
import argparse, sys, time, numpy as np, torch
import _paths  # noqa: E402  (portable path bootstrap)
torch.set_grad_enabled(False)
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
from generic_runtime_wired import DecoderLayer, _bucket
from generic_runtime import ModelConfig, GenericRuntime, rmsnorm

BF = torch.bfloat16
def dv(a, dev): return torch.from_numpy(np.ascontiguousarray(a)).to(BF).to(dev)
def dvf(a, dev): return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32).to(dev)


class ChipRuntime:
    """One wired-decode module bound to a physical chip. Weights resident on that chip."""
    def __init__(self, cfg, sd, S, chip):
        self.c = cfg; self.S = S; self.inv = cfg.inv_freq(); self.dev = f"rngd:{chip}"
        g = lambda k: sd[k].astype(np.float32)
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.lm = self.embed if cfg.tie else g("lm_head.weight")
        hd = cfg.head_dim; H = cfg.hidden
        self.hd = hd; self.NH = cfg.n_heads; self.NKV = cfg.n_kv
        self.Hq = cfg.n_heads*hd; self.Hkv = cfg.n_kv*hd; self.qkv_out = self.Hq+2*self.Hkv
        half = hd//2; z = lambda *s: torch.zeros(*s, dtype=BF)
        sample = (torch.zeros(H, dtype=torch.float32), z(H), z(self.qkv_out, H), z(H, self.Hq), z(H),
                  z(2*cfg.inter, H), z(H, cfg.inter), z(self.NKV, S, hd), z(self.NKV, S, hd),
                  torch.zeros(half, dtype=torch.float32), torch.zeros(half, dtype=torch.float32),
                  torch.zeros(S, dtype=torch.float32), torch.zeros(S, dtype=torch.float32))
        t0 = time.time()
        self.m = CompileModule.from_module(DecoderLayer(cfg, S), sample); self.m.to(self.dev)
        self.compile_s = time.time()-t0
        self.L = []
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            wqkv = np.concatenate([g(p+"self_attn.q_proj.weight"), g(p+"self_attn.k_proj.weight"), g(p+"self_attn.v_proj.weight")], 0)
            wgu = np.concatenate([g(p+"mlp.gate_proj.weight"), g(p+"mlp.up_proj.weight")], 0)
            self.L.append(dict(n1=dv(g(p+"input_layernorm.weight"), self.dev), n2=dv(g(p+"post_attention_layernorm.weight"), self.dev),
                               wqkv=dv(wqkv, self.dev), wo=dv(g(p+"self_attn.o_proj.weight"), self.dev),
                               wgu=dv(wgu, self.dev), wd=dv(g(p+"mlp.down_proj.weight"), self.dev)))

    def step(self, tok, pos, ck, cv):
        """One token; updates ck/cv (host numpy KV) in place. Returns argmax."""
        c = self.c; hd = self.hd; S = self.S
        ang = pos*self.inv; cos = dvf(np.cos(ang), self.dev); sin = dvf(np.sin(ang), self.dev)
        mask = np.full(S, -1e4, np.float32); mask[:pos+1] = 0.0
        oh = np.zeros(S, np.float32); oh[pos] = 1.0
        x = dvf(self.embed[tok].astype(np.float32), self.dev)
        for i in range(c.n_layers):
            w = self.L[i]
            hid, k, v = self.m(x, w["n1"], w["wqkv"], w["wo"], w["n2"], w["wgu"], w["wd"],
                               dv(ck[i], self.dev), dv(cv[i], self.dev), cos, sin, dvf(mask, self.dev), dvf(oh, self.dev))
            x = hid
            if pos < S:
                ck[i][:, pos, :] = k.detach().float().cpu().numpy().reshape(self.NKV, hd)
                cv[i][:, pos, :] = v.detach().float().cpu().numpy().reshape(self.NKV, hd)
        return int(np.argmax(self.lm @ rmsnorm(x.detach().float().cpu().numpy(), self.norm, c.eps)))

    def new_kv(self):
        z = lambda: [np.zeros((self.NKV, self.S, self.hd), np.float32) for _ in range(self.c.n_layers)]
        return z(), z()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--split", type=int, default=3)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd); seq = ids + oracle.generate(ids, a.n)
    S = _bucket(len(seq)); P = min(a.split, len(seq)-1)
    print(f"[pd-2chip] compiling prefill@rngd:0 + decode@rngd:1 (S={S})...", flush=True)
    rt0 = ChipRuntime(cfg, sd, S, chip=0)   # PREFILL chip
    rt1 = ChipRuntime(cfg, sd, S, chip=1)   # DECODE chip
    print(f"  compiled: chip0 {rt0.compile_s:.0f}s, chip1 {rt1.compile_s:.0f}s", flush=True)

    # monolithic reference (one chip)
    ckM, cvM = rt0.new_kv(); mono = [rt0.step(seq[p], p, ckM, cvM) for p in range(len(seq))]

    # ---- REAL PD ----
    # PREFILL on chip0 → produces KV (ck0/cv0, host numpy)
    ck0, cv0 = rt0.new_kv()
    t0 = time.time(); pre = [rt0.step(seq[p], p, ck0, cv0) for p in range(P)]; t_pre = time.time()-t0
    # KV HANDOFF chip0 → host → chip1 (ck0/cv0 already on host; measure size + the re-materialize)
    t0 = time.time()
    kv_bytes = sum(a.nbytes for a in ck0) + sum(a.nbytes for a in cv0)
    ck1 = [a.copy() for a in ck0]; cv1 = [a.copy() for a in cv0]   # the tensors that cross to chip1
    t_move = time.time()-t0
    # DECODE on chip1 from the handed-off KV
    t0 = time.time(); dec = [rt1.step(seq[p], p, ck1, cv1) for p in range(P, len(seq))]; t_dec = time.time()-t0
    pd = pre + dec

    ok = pd == mono
    print("\n" + "=" * 62)
    print(f"REAL cross-chip PD — prefill@rngd:0 (0..{P-1}) → KV → decode@rngd:1 ({P}..)")
    print(f"  KV moved chip0→chip1: {kv_bytes/1e6:.1f} MB (padded S={S})")
    print(f"  prefill@chip0: {t_pre/P*1e3:.0f} ms/tok | decode@chip1: {t_dec/(len(seq)-P)*1e3:.0f} ms/tok")
    print(f"  monolithic: {mono}")
    print(f"  PD 2-chip:  {pd}")
    print(f"  {'PASS ✅ KV genuinely crossed chips, decode used it, output identical' if ok else 'FAIL ❌'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
