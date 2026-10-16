"""Port the 4 DSL kernels (rms/zipadd/swiglu/softmax) to the 2026.3 tcl eDSL.

Validates numerics on the CPU graph-calculator target, then dumps the
generated DSL text so we can see the new grammar.
"""
from __future__ import annotations

# ruff: noqa: F821  (B, N, K, G, S are symbolic axes parsed from annotations)

import torch
import furiosa.tcl as tcl
from furiosa.tcl import commit, fetch, operation

EPS_H = 1e-5 * 4096          # matches the old kernels' `e` tensor (EPS*H)


# --- tensor operations -------------------------------------------------
@tcl.tensor_operation
def _rms(x, w, eps, axes):
    """x / sqrt(sum(x^2) + eps) * w   — bit-semantics of the old rms_dsl."""
    xf = fetch.typecast(x, tcl.f32)
    sq = operation.mulf(xf, xf)
    s = operation.reduce(sq, axes=axes, op=tcl.Addf)
    s = operation.addf(s, eps)
    r = operation.sqrt(s)
    o = operation.divf(xf, r)
    o = operation.mulf(o, w)
    return commit.typecast(o, tcl.bf16)


@tcl.tensor_operation(tactic_kind=tcl.Interleaving)
def _zipadd(a, b):
    return operation.addf(a, b)


@tcl.tensor_operation(tactic_kind=tcl.Elementwise)
def _silu(g):
    s = operation.sigmoid(g)
    return operation.mulf(s, g)


@tcl.tensor_operation(tactic_kind=tcl.Interleaving)
def _mul(a, b):
    return operation.mulf(a, b)


@tcl.tensor_operation
def _softmax(x, axes):
    xf = fetch.typecast(x, tcl.f32)
    m = operation.reduce(xf, axes=axes, op=tcl.Maxf)
    c = operation.subf(xf, m)
    e = operation.exp(c)
    d = operation.reduce(e, axes=axes, op=tcl.Addf)
    o = operation.divf(e, d)
    return commit.typecast(o, tcl.bf16)


# --- kernels -----------------------------------------------------------
@tcl.kernel
def rms_kernel(
    x: tcl.Tensor[tcl.bf16, [B, N]],
    w: tcl.Tensor[tcl.bf16, [B, N]],
) -> tcl.Tensor[tcl.bf16, [B, N]]:
    out: tcl.Tensor[tcl.bf16, [B, N]] = _rms(x, w, EPS_H, [N])
    return out


@tcl.kernel
def zipadd_kernel(
    a: tcl.Tensor[tcl.bf16, [B, N]],
    b: tcl.Tensor[tcl.bf16, [B, N]],
) -> tcl.Tensor[tcl.bf16, [B, N]]:
    out: tcl.Tensor[tcl.bf16, [B, N]] = _zipadd(a, b)
    return out


@tcl.kernel
def swiglu_kernel(
    g: tcl.Tensor[tcl.bf16, [B, N]],
    u: tcl.Tensor[tcl.bf16, [B, N]],
) -> tcl.Tensor[tcl.bf16, [B, N]]:
    sg: tcl.Tensor[tcl.bf16, [B, N]] = _silu(g)
    out: tcl.Tensor[tcl.bf16, [B, N]] = _mul(sg, u)
    return out


@tcl.kernel
def softmax_kernel(
    sc: tcl.Tensor[tcl.bf16, [B, K, G, S]],
) -> tcl.Tensor[tcl.bf16, [B, K, G, S]]:
    out: tcl.Tensor[tcl.bf16, [B, K, G, S]] = _softmax(sc, [S])
    return out


# --- checks ------------------------------------------------------------
def _close(got, ref, name, rtol=5e-2, atol=1e-2):
    err = (got.float() - ref.float()).abs().max().item()
    ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
    print(f"  {name:10s} max|err|={err:.5f}  {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    from furiosa.torch import TclModule
    torch.manual_seed(0)
    bf = torch.bfloat16
    B, N, S = 4, 4096, 128
    allok = True

    x = (torch.randn(B, N) * 0.1).to(bf)
    w = torch.randn(B, N).to(bf)
    got = TclModule(rms_kernel)(x, w)[0]
    ref = x.float() / (x.float().pow(2).sum(-1, keepdim=True) + EPS_H).sqrt() * w.float()
    allok &= _close(got, ref.to(bf), "rms")

    a, b = (torch.randn(B, N) * 0.1).to(bf), (torch.randn(B, N) * 0.1).to(bf)
    allok &= _close(TclModule(zipadd_kernel)(a, b)[0], (a.float() + b.float()).to(bf), "zipadd")

    g, u = (torch.randn(B, 14336) * 0.5).to(bf), (torch.randn(B, 14336) * 0.5).to(bf)
    allok &= _close(TclModule(swiglu_kernel)(g, u)[0],
                    (torch.nn.functional.silu(g.float()) * u.float()).to(bf), "swiglu")

    sc = (torch.randn(B, 8, 4, S) * 2).to(bf)
    allok &= _close(TclModule(softmax_kernel)(sc)[0],
                    torch.softmax(sc.float(), dim=-1).to(bf), "softmax")

    print("\n=== generated DSL text (new grammar), rms_kernel ===")
    print(tcl.Kernel.from_edsl(rms_kernel).dsl)
    print("ALL OK" if allok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
