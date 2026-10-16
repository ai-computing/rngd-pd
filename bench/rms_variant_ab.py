#!/usr/bin/env python3
"""Two spellings of the same RMSNorm, timed against each other.

The kernel this repo has always used computes 1/sqrt(s) as exp(-0.5*log(s)),
which costs two extra VE instructions and -- because the -0.5 arrives as a
tensor -- two extra [B,H] f32 reads per call. rms runs twice a layer, so at 32
layers that is 8 MB a step of operands that could be literals. 2026.3's eDSL
emits sqrt/div directly, which is what suggested the comparison; both forms are
spelled here in the text DSL so the same file runs on both SDKs.

The sqrt form keeps the 4-argument signature (eps still arrives as a tensor) so
that swapping it in costs nothing at the call sites.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from rngd_pd.engines.fsdk import open_device  # noqa: E402

open_device(8, 2)

from rngd_pd.engines.fsdk import (SDK, CompileModule, FCConfig,  # noqa: E402
                                  TacticHintConfig, TacticKernelModule,
                                  dev_str, sync)

H = 4096
BF = torch.bfloat16

LOGEXP = """#tactic_kernel_dsl
def rms(x: [B={B},N={H}]/bf16, w: [B={B},N={H}]/bf16, e: [B={B},N={H}]/f32, nh: [B={B},N={H}]/f32) -> [B={B},N={H}]/bf16:
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

SQRT = """#tactic_kernel_dsl
def rms(x: [B={B},N={H}]/bf16, w: [B={B},N={H}]/bf16, e: [B={B},N={H}]/f32, nh: [B={B},N={H}]/f32) -> [B={B},N={H}]/bf16:
    output: [B,N]/bf16 = tk.ReduceByVe(
        %0 = read(x)
        %1 = read(w)
        %2 = read(e)
        %3 = ve.exec(%0 * %0)
        %4 = ve.exec(+ t[N] %3)
        %5 = ve.exec(%4 + %2)
        %6 = ve.exec(sqrt %5)
        %7 = ve.exec(%0 / %6)
        %8 = ve.exec(%7 * %1)
        write.to([B,N]/bf16)
    )
    return output
"""

VARIANTS = {"logexp": LOGEXP, "sqrt": SQRT}


class Rms(torch.nn.Module):
    def __init__(self, dsl):
        super().__init__()
        self.k = TacticKernelModule(dsl)

    def forward(self, x, w, e, nh):
        return self.k(x, w, e, nh)[0]


def sample(B):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(B, H, generator=g).mul(0.1).to(BF),
            torch.randn(B, H, generator=g).to(BF),
            torch.full((B, H), 1e-5 * H),
            torch.full((B, H), -0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="1,4,8,16")
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--hint", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    hint = a.hint or ("Default" if hasattr(TacticHintConfig, "ForLlmModelIOBound")
                      else "ForLlmModelDecode")
    cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                   tactic_hint=getattr(TacticHintConfig, hint))
    dev = dev_str(a.chip)
    rows = []
    print(f"[rms-ab] SDK={SDK} hint={hint} chip={a.chip}", flush=True)
    print(f"{'B':>3} {'variant':>8} {'us/call':>9} {'max|err|':>9}", flush=True)
    for B in [int(x) for x in a.batch.split(",")]:
        host = sample(B)
        x, w, e, _ = host
        ref = (x.float() / (x.float().pow(2).sum(-1, keepdim=True) + 1e-5 * H)
               .sqrt() * w.float())
        dv = [t.to(dev) for t in host]
        for name, tmpl in VARIANTS.items():
            try:
                m = CompileModule.from_module(
                    Rms(tmpl.format(B=B, H=H)), host, compiler_config=cfg)
                m.to(dev)
                out = m(*dv)
                got = (out[0] if isinstance(out, (tuple, list)) else out)
                err = (got.detach().cpu().float() - ref).abs().max().item()
                sync(a.chip)
                ts = []
                for _ in range(a.iters):
                    t = time.perf_counter()
                    m(*dv)
                    sync(a.chip)
                    ts.append(time.perf_counter() - t)
                ts.sort()
                us = ts[len(ts) // 2] * 1e6
                rows.append(dict(sdk=SDK, B=B, variant=name, us=round(us, 1),
                                 err=round(err, 5), hint=hint))
                print(f"{B:3d} {name:>8} {us:9.1f} {err:9.5f}", flush=True)
                del m
            except Exception as ex:
                rows.append(dict(sdk=SDK, B=B, variant=name,
                                 error=f"{type(ex).__name__}: {str(ex)[:160]}"))
                print(f"{B:3d} {name:>8}      FAIL  {type(ex).__name__}: "
                      f"{str(ex)[:90]}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"-> {a.out}", flush=True)
    for B in sorted({r["B"] for r in rows}):
        d = {r["variant"]: r.get("us") for r in rows if r["B"] == B}
        if d.get("logexp") and d.get("sqrt"):
            print(f"B={B}: sqrt is {d['logexp']/d['sqrt']:.2f}x logexp",
                  flush=True)


if __name__ == "__main__":
    main()
