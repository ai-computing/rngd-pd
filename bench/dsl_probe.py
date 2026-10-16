#!/usr/bin/env python3
"""Which DSL kernel shape makes the tactic populator panic?

The batched decode engine dies in the compiler at
`tactic_kernel/reduce_by_ve.rs`, and both reduce kernels — rms_norm (sum over
the hidden dim) and softmax (max/sum over the key dim) — grow a leading batch
dimension in the batched variants. The elementwise kernels (zip_add, swiglu)
have no reduce at all, so this isolates the reduce shapes one at a time and
tells us whether the batch dim itself is the problem or only some sizes of it.

Also checks the shape a chunked prefill would want: C positions folded into the
existing group dimension rather than added as a new leading one, which keeps the
reduce over the last axis exactly where the working kernel already has it.
"""
import argparse
import time

import torch

import _bootstrap  # noqa: F401
import furiosa.torch  # noqa: F401
from rngd_pd.engines.fsdk import open_device as _open_device

_open_device(8)
from rngd_pd.engines.fsdk import TacticKernelModule, CompileModule  # noqa: E402
from rngd_pd.engines.fsdk import FCConfig, TacticHintConfig  # noqa: E402

BF = torch.bfloat16


def rms_flat(N):
    return f"""#tactic_kernel_dsl
def rms_norm(x: [N={N}]/bf16, w: [N={N}]/bf16, eps: [N={N}]/f32, nh: [N={N}]/f32) -> [N={N}]/bf16:
    output: [N]/bf16 = tk.ReduceByVe(
        %0 = read(x)
        %1 = read(w)
        %2 = read(eps)
        %3 = read(nh)
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


def rms_batched(B, N):
    return f"""#tactic_kernel_dsl
def rms_norm(x: [B={B},N={N}]/bf16, w: [B={B},N={N}]/bf16, eps: [B={B},N={N}]/f32, nh: [B={B},N={N}]/f32) -> [B={B},N={N}]/bf16:
    output: [B,N]/bf16 = tk.ReduceByVe(
        %0 = read(x)
        %1 = read(w)
        %2 = read(eps)
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


def softmax_flat(K, G, S):
    return f"""#tactic_kernel_dsl
def softmax_attn(sc: [K={K},G={G},S={S}]/bf16) -> [K={K},G={G},S={S}]/bf16:
    output: [K,G,S]/bf16 = tk.ReduceByVe(
        %0 = read(sc)
        %1 = ve.exec(max_f t[S] %0)
        %2 = ve.exec(%0 - %1)
        %3 = ve.exec(exp %2)
        %4 = ve.exec(+ t[S] %3)
        %5 = ve.exec(%3 / %4)
        write.to([K,G,S]/bf16)
    )
    return output
"""


def softmax_batched(B, K, G, S):
    return f"""#tactic_kernel_dsl
def softmax_attn(sc: [B={B},K={K},G={G},S={S}]/bf16) -> [B={B},K={K},G={G},S={S}]/bf16:
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


class One(torch.nn.Module):
    def __init__(self, dsl):
        super().__init__()
        self.k = TacticKernelModule(dsl)

    def forward(self, *args):
        return self.k(*args)


def check(name, dsl, sample, dev):
    cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                   tactic_hint=TacticHintConfig.ForLlmModelDecode)
    t0 = time.time()
    try:
        m = CompileModule.from_module(One(dsl), sample, compiler_config=cfg)
        m.to(dev)
        m(*[s.to(dev) for s in sample])
        print(f"  OK   {name}  ({time.time()-t0:.0f}s)", flush=True)
        return True
    except BaseException as e:
        print(f"  FAIL {name}: {type(e).__name__} {str(e)[:70]}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, default=2)  # npu3 has the EDF loader bug
    ap.add_argument("--H", type=int, default=4096)
    a = ap.parse_args()
    dev = f"rngd:{a.chip}"
    H = a.H
    z = lambda *s: torch.zeros(*s, dtype=BF)
    f = lambda *s: torch.zeros(*s, dtype=torch.float32)

    print("rms_norm — reduce over the hidden dim")
    check("flat [N]", rms_flat(H), (z(H), z(H), f(H), f(H)), dev)
    for B in (1, 2, 4):
        check(f"batched [B={B},N]", rms_batched(B, H),
              (z(B, H), z(B, H), f(B, H), f(B, H)), dev)

    print("\nsoftmax — reduce over the key dim")
    check("flat [K,G,S=64]", softmax_flat(8, 4, 64), (z(8, 4, 64),), dev)
    for B in (1, 2):
        check(f"batched [B={B},K,G,S=64]", softmax_batched(B, 8, 4, 64),
              (z(B, 8, 4, 64),), dev)

    print("\nchunked-prefill shape — C folded into the group dim, no new axis")
    for C in (4, 16):
        check(f"flat [K,G*C={4*C},S=64+C]", softmax_flat(8, 4 * C, 64 + C),
              (z(8, 4 * C, 64 + C),), dev)


if __name__ == "__main__":
    main()
