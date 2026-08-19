"""Batched wired Llama-8B decode — realistic serving: batch amortizes the memory-bound weight read.

Extends llama_engine.py with a batch dim B. Decode reads all weights ONCE per step regardless of B,
so tok/s ≈ B × single-batch until compute-bound (ridge ~171). This is THE realistic decode metric
(batch=1 was unrealistic). DSL kernels take [B,N]; matmuls batch over B; KV is per-sequence [B,...].

Sweep batch: python3 llama_engine_batch.py --nl 8 --tactic Decode --batch 1,8,32,64
"""
import torch, furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
import sys, math, time, argparse
try: import _tv_patch  # noqa
except Exception: pass
from furiosa.torch.custom_ops import TacticKernelModule, CompileModule
from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig

H = 4096; M = 14336; NUM_HEADS = 32; NUM_KV = 8; HEAD_DIM = 128
GQA_GROUP = NUM_HEADS // NUM_KV; EPS = 1e-5; NL = 32


def rms_dsl(B):
    return f"""#tactic_kernel_dsl
def rms(x: [B={B},N=4096]/bf16, w: [B={B},N=4096]/bf16, e: [B={B},N=4096]/f32, nh: [B={B},N=4096]/f32) -> [B={B},N=4096]/bf16:
    output: [B,N]/bf16 = tk.ReduceByVe(
        %0 = read(x)
        %1 = read(w)
        %2 = read(e)
        %3 = read(nh)
        %4 = ve.exec(%0 * %0)
        %5 = ve.exec(+ t[N] %4)
        %6 = ve.exec(%5 + %2)
        %7 = ve.exec(log %6)
        %8 = ve.exec(%7 * %3)
        %9 = ve.exec(exp %8)
        %10 = ve.exec(%0 * %9)
        %11 = ve.exec(%10 * %1)
        write.to([B,N]/bf16)
    )
    return output
"""

def zipadd_dsl(B):
    return f"""#tactic_kernel_dsl
def zipadd(a: [B={B},N=4096]/bf16, b: [B={B},N=4096]/bf16) -> [B={B},N=4096]/bf16:
    output: [B,N]/bf16 = tk.Interleaving(
        %0 = read(a)
        %1 = read(b)
        %2 = ve.exec(%0 + %1)
        write.to([B,N]/bf16)
    )
    return output
"""

def swiglu_dsl(B):
    return f"""#tactic_kernel_dsl
def swiglu(g: [B={B},N=14336]/bf16, u: [B={B},N=14336]/bf16) -> [B={B},N=14336]/bf16:
    sg: [B,N]/bf16 = tk.Elementwise(
        %0 = read(g)
        %1 = ve.exec(sigmoid %0)
        %2 = ve.exec(%0 * %1)
        write.to([B,N]/bf16)
    )
    output: [B,N]/bf16 = tk.Interleaving(
        %0 = read(sg)
        %1 = read(u)
        %2 = ve.exec(%0 * %1)
        write.to([B,N]/bf16)
    )
    return output
"""

def softmax_dsl(B, S):
    return f"""#tactic_kernel_dsl
def softmax(sc: [B={B},K=8,G=4,S={S}]/bf16) -> [B={B},K=8,G=4,S={S}]/bf16:
    output: [B,K,G,S]/bf16 = tk.ReduceByVe(
        %0 = read(sc)
        %1 = ve.exec(max_f t[S] %0)
        %2 = ve.exec(%0 - %1)
        %3 = ve.exec(exp %2)
        %4 = ve.exec(+ t[S] %3)
        %5 = ve.exec(%3 / %4)
        write.to([B,K,G,S]/bf16)
    )
    return output
"""


class BLayer(torch.nn.Module):
    def __init__(self, B, S):
        super().__init__(); self.B = B; self.S = S
        self.rms1 = TacticKernelModule(rms_dsl(B)); self.rms2 = TacticKernelModule(rms_dsl(B))
        self.sm = TacticKernelModule(softmax_dsl(B, S))
        self.r1 = TacticKernelModule(zipadd_dsl(B)); self.r2 = TacticKernelModule(zipadd_dsl(B))
        self.sw = TacticKernelModule(swiglu_dsl(B))
    def forward(self, x, n1, e, nh, wq, wo, kp, vp, n2, wg, wu, wd):
        B = self.B
        h = self.rms1(x, n1, e, nh)[0]                       # [B,H]
        q = torch.matmul(h, wq.t()).reshape(B, NUM_KV, GQA_GROUP, HEAD_DIM)   # [B,K,G,hd]
        sc = torch.matmul(q, kp)                             # [B,K,G,S]  kp [B,K,hd,S]
        a = self.sm(sc)[0]
        ao = torch.matmul(a, vp).reshape(B, NUM_HEADS*HEAD_DIM)   # vp [B,K,S,hd] → [B,K,G,hd] → [B,Hq]
        o = torch.matmul(ao, wo.t())                         # [B,H]
        h2 = self.r1(o, x)[0]
        n = self.rms2(h2, n2, e, nh)[0]
        gg = torch.matmul(n, wg.t()); uu = torch.matmul(n, wu.t())   # [B,M]
        gated = self.sw(gg, uu)[0]
        return self.r2(torch.matmul(gated, wd.t()), h2)[0]


class BModel(torch.nn.Module):
    def __init__(self, B, S, nl):
        super().__init__(); self.nl = nl
        self.layers = torch.nn.ModuleList([BLayer(B, S) for _ in range(nl)])
    def forward(self, x, e, nh, an1, aq, ao, ak, av, an2, ag, au, ad):
        h = x
        for L in range(self.nl):
            h = self.layers[L](h, an1[L], e, nh, aq[L], ao[L], ak[L], av[L], an2[L], ag[L], au[L], ad[L])
        return h


def make_sample(B, S, nl):
    torch.manual_seed(0); bf = torch.bfloat16
    x = (torch.randn(B, H)*0.1).to(bf)
    e = torch.full((B, H), EPS*H, dtype=torch.float32); nh = torch.full((B, H), -0.5, dtype=torch.float32)
    r = lambda *s: (torch.randn(*s)*0.02).to(bf)
    an1 = (torch.randn(nl, B, H)).to(bf); an2 = (torch.randn(nl, B, H)).to(bf)
    aq = r(nl, NUM_HEADS*HEAD_DIM, H); ao = r(nl, H, NUM_HEADS*HEAD_DIM)
    ak = (torch.randn(nl, B, NUM_KV, HEAD_DIM, S)*0.1).to(bf); av = (torch.randn(nl, B, NUM_KV, S, HEAD_DIM)*0.1).to(bf)
    ag = r(nl, M, H); au = r(nl, M, H); ad = r(nl, H, M)
    return (x, e, nh, an1, aq, ao, ak, av, an2, ag, au, ad)


def bench(B, S, nl, tactic, chip, secs=0, label=""):
    DEV = f"rngd:{chip}"
    cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                   tactic_hint=TacticHintConfig.ForLlmModelDecode) if tactic == "Decode" else None
    model = BModel(B, S, nl); sample = make_sample(B, S, nl)
    m = CompileModule.from_module(model, sample, compiler_config=cfg); m.to(DEV)
    dv = tuple(t.to(DEV) for t in sample)
    for _ in range(5): m(*dv)
    if secs > 0:   # loop for a fixed WALL time (for concurrent overlap) — sync via ready file
        rf = f"/tmp/pd_ready_chip{chip}"; open(rf, "w").write("ready")
        while not all(__import__("os").path.exists(f"/tmp/pd_ready_chip{c}") for c in (0, 1)):
            time.sleep(0.2)   # barrier: both chips start timing together
        t0 = time.time(); n = 0
        while time.time() - t0 < secs: m(*dv); n += 1
        ms = (time.time()-t0)/n*1000
    else:
        t0 = time.time()
        for _ in range(30): m(*dv)
        ms = (time.time()-t0)/30*1000
    wbytes = nl*(4096*4096*2*2 + 14336*4096*2*3)   # weights read ONCE (shared across batch)
    flop = B * nl * (2*(4096*4096*2 + 14336*4096*3))  # ~2*params*B
    tok_s = B/(ms/1000)
    print(f"  {label}B={B:3d} chip{chip}: {ms:.2f} ms/step | {tok_s:.0f} tok/s | wBW {wbytes/(ms/1000)/1e9:.0f} GB/s | compute {flop/(ms/1000)/1e12:.1f} TFLOPS", flush=True)
    return B, ms, tok_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", type=int, default=8); ap.add_argument("--bucket", type=int, default=128)
    ap.add_argument("--tactic", default="Decode"); ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--batch", default="1,8,32,64")
    ap.add_argument("--secs", type=int, default=0, help="loop for fixed wall time (concurrent PD)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    print(f"[batch] nl={a.nl} S={a.bucket} tactic={a.tactic} chip={a.chip} batch={a.batch} secs={a.secs}", flush=True)
    for B in [int(x) for x in a.batch.split(",")]:
        try: bench(B, a.bucket, a.nl, a.tactic, a.chip, a.secs, a.label)
        except Exception as ex: print(f"  B={B}: FAIL {str(ex)[:100]}", flush=True)


if __name__ == "__main__":
    main()
