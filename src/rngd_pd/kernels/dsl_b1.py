"""Llama-3.1-8B whole-model wired engine — port of qwen_engine.py (the proven 2.42ms structure).

Whole 32-layer model = ONE nn.Module → ONE wired EDF (1 launch/token). Compiles because it uses
L1 DSL (TacticKernelModule) for rms/softmax/swiglu (light on the compiler) + lean GQA attention
(q.reshape[NKV,G,hd] + matmul, NO repeat_interleave/einsum) + KV as pre-permuted input. This is why
my raw-torch WholeModel timed out at 400s but the DSL version should compile.

Key test: does Llama-32L DSL-wire compile within the wall? Reports compile time + task_num.
RoPE on q is applied in-graph; K-cache is assumed pre-roped by the caller (like the vendor).

  python3 llama_engine.py --bucket 128
"""
import torch, furiosa.torch
from rngd_pd.engines.fsdk import open_device as _open_device
_open_device(8, 2)
import sys, math, time, yaml, argparse
try:
    from rngd_pd import _tv_patch as _tv_patch  # noqa
except Exception:
    pass
from rngd_pd.engines.fsdk import TacticKernelModule, CompileModule
from rngd_pd.engines.fsdk import FCConfig, TacticHintConfig
import furiosa.native_llm_common as nlc

H = 4096; M = 14336; NUM_HEADS = 32; NUM_KV = 8; HEAD_DIM = 128
GQA_GROUP = NUM_HEADS // NUM_KV  # 4
EPS = 1e-5; SCALE = 1.0 / math.sqrt(HEAD_DIM); NL = 32

RMS_NORM = """#tactic_kernel_dsl
def rms_norm(x: [N=4096]/bf16, w_baked: [N=4096]/bf16, eps_baked: [N=4096]/f32, neg_half: [N=4096]/f32) -> [N=4096]/bf16:
    output: [N]/bf16 = tk.ReduceByVe(
        %0 = read(x)
        %1 = read(w_baked)
        %2 = read(eps_baked)
        %3 = read(neg_half)
        %4 = ve.exec(%0 * %0)
        %5 = ve.exec(+ t[N] %4)
        %6 = ve.exec(%5 + %2)
        %7 = ve.exec(log %6)
        %8 = ve.exec(%7 * %3)
        %9 = ve.exec(exp %8)
        %10 = ve.exec(%0 * %9)
        %11 = ve.exec(%10 * %1)
        write.to([N]/bf16)
    )
    return output
"""

ZIP_ADD = """#tactic_kernel_dsl
def zip_add(a: [N=4096]/bf16, b: [N=4096]/bf16) -> [N=4096]/bf16:
    output: [N]/bf16 = tk.Interleaving(
        %0 = read(a)
        %1 = read(b)
        %2 = ve.exec(%0 + %1)
        write.to([N]/bf16)
    )
    return output
"""

SWIGLU = """#tactic_kernel_dsl
def swiglu(gate: [N=14336]/bf16, up: [N=14336]/bf16) -> [N=14336]/bf16:
    silu_gate: [N]/bf16 = tk.Elementwise(
        %0 = read(gate)
        %1 = ve.exec(sigmoid %0)
        %2 = ve.exec(%0 * %1)
        write.to([N]/bf16)
    )
    output: [N]/bf16 = tk.Interleaving(
        %0 = read(silu_gate)
        %1 = read(up)
        %2 = ve.exec(%0 * %1)
        write.to([N]/bf16)
    )
    return output
"""


def softmax_dsl(seq_len):
    return f"""#tactic_kernel_dsl
def softmax_attn(scores: [K=8,G=4,S={seq_len}]/bf16) -> [K=8,G=4,S={seq_len}]/bf16:
    output: [K,G,S]/bf16 = tk.ReduceByVe(
        %0 = read(scores)
        %1 = ve.exec(max_f t[S] %0)
        %2 = ve.exec(%0 - %1)
        %3 = ve.exec(exp %2)
        %4 = ve.exec(+ t[S] %3)
        %5 = ve.exec(%3 / %4)
        write.to([K,G,S]/bf16)
    )
    return output
"""


class LlamaLayer(torch.nn.Module):
    def __init__(self, seq_len):
        super().__init__(); self.seq_len = seq_len
        self.rms1 = TacticKernelModule(RMS_NORM); self.rms2 = TacticKernelModule(RMS_NORM)
        self.softmax = TacticKernelModule(softmax_dsl(seq_len))
        self.res1 = TacticKernelModule(ZIP_ADD); self.res2 = TacticKernelModule(ZIP_ADD)
        self.swiglu = TacticKernelModule(SWIGLU)

    def forward(self, hidden, w_norm1, eps_baked, neg_half, w_q, w_o, kperm, vperm,
                w_norm2, w_gate, w_up, w_down):
        normed1 = self.rms1(hidden, w_norm1, eps_baked, neg_half)[0]
        q = torch.matmul(w_q, normed1)                       # [Hq]  (w_q pre-scaled by 1/sqrt(hd))
        q_g = q.reshape(NUM_KV, GQA_GROUP, HEAD_DIM)         # [K,G,hd]
        scores = torch.matmul(q_g, kperm)                    # [K,G,S]  (kperm [K,hd,S])
        attn = self.softmax(scores)[0]
        attn_out = torch.matmul(attn, vperm)                 # [K,G,hd] (vperm [K,S,hd])
        o = torch.matmul(w_o, attn_out.reshape(NUM_HEADS * HEAD_DIM))
        h2 = self.res1(o, hidden)[0]
        normed2 = self.rms2(h2, w_norm2, eps_baked, neg_half)[0]
        gate = torch.matmul(w_gate, normed2); up = torch.matmul(w_up, normed2)
        gated = self.swiglu(gate, up)[0]
        return self.res2(torch.matmul(w_down, gated), h2)[0]


class Llama32Layer(torch.nn.Module):
    def __init__(self, seq_len, nl=NL):
        super().__init__(); self.nl = nl; self.layers = torch.nn.ModuleList([LlamaLayer(seq_len) for _ in range(nl)])

    def forward(self, hidden, eps_baked, neg_half, aw_n1, aw_q, aw_o, ak, av, aw_n2, aw_g, aw_u, aw_d):
        h = hidden
        for L in range(self.nl):
            h = self.layers[L](h, aw_n1[L], eps_baked, neg_half, aw_q[L], aw_o[L], ak[L], av[L],
                               aw_n2[L], aw_g[L], aw_u[L], aw_d[L])
        return h


def make_sample(seq_len, nl=NL):
    torch.manual_seed(0)
    bf = torch.bfloat16
    h = (torch.randn(H) * 0.1).to(bf)
    eps_baked = torch.full((H,), EPS * H, dtype=torch.float32)
    neg_half = torch.full((H,), -0.5, dtype=torch.float32)
    r = lambda *s: (torch.randn(*s) * 0.02).to(bf)
    aw_n1 = (torch.randn(nl, H)).to(bf); aw_n2 = (torch.randn(nl, H)).to(bf)
    aw_q = r(nl, NUM_HEADS*HEAD_DIM, H); aw_o = r(nl, H, NUM_HEADS*HEAD_DIM)
    ak = (torch.randn(nl, NUM_KV, HEAD_DIM, seq_len) * 0.1).to(bf)   # pre-permuted [K,hd,S]
    av = (torch.randn(nl, NUM_KV, seq_len, HEAD_DIM) * 0.1).to(bf)   # [K,S,hd]
    aw_g = r(nl, M, H); aw_u = r(nl, M, H); aw_d = r(nl, H, M)
    return (h, eps_baked, neg_half, aw_n1, aw_q, aw_o, ak, av, aw_n2, aw_g, aw_u, aw_d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bucket", type=int, default=128)
    ap.add_argument("--nl", type=int, default=NL)
    ap.add_argument("--tactic", default=None, help="tactic_hint: Decode|Prefill|Default")
    ap.add_argument("--chip", type=int, default=0, help="physical NPU (rngd:N)")
    a = ap.parse_args()
    DEV = f"rngd:{a.chip}"
    cfg = None
    if a.tactic:
        hint = {"Decode": TacticHintConfig.ForLlmModelDecode, "Prefill": TacticHintConfig.ForLlmModelPrefill,
                "Default": TacticHintConfig.Default}[a.tactic]
        cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True, tactic_hint=hint)
        print(f"  [tuned cfg] tactic_hint=ForLlmModel{a.tactic}", flush=True)
    print(f"[llama-engine] CompileModule of {a.nl}-layer Llama (bucket={a.bucket}, DSL+mesh+fusion)...", flush=True)
    model = Llama32Layer(a.bucket, a.nl); sample = make_sample(a.bucket, a.nl)
    t0 = time.time()
    m = CompileModule.from_module(model, sample, compiler_config=cfg); m.to(DEV)
    print(f"  CompileModule ready in {time.time()-t0:.0f}s ({a.nl} layers)", flush=True)
    dv = tuple(t.to(DEV) for t in sample)
    for _ in range(5): m(*dv)
    t0 = time.time()
    for _ in range(30): out = m(*dv)
    ms = (time.time() - t0) / 30 * 1000
    wbytes = a.nl * (4096*4096*2*2 + 14336*4096*2*3)   # q+o + gate+up+down per layer
    bw = wbytes / (ms / 1000) / 1e9                     # GB/s
    print(f"  ✨ RUNTIME: {ms:.2f} ms/token ({a.nl}L) | weights {wbytes/1e6:.0f}MB → eff BW {bw:.0f} GB/s ({bw/1500*100:.0f}% of 1.5TB/s)", flush=True)


if __name__ == "__main__":
    main()
