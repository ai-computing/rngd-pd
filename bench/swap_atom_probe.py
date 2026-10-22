"""ATOM: can separately compiled graphs share weights that are graph inputs? (#17)

Public API only (rebel-compiler, torch-rbln). A toy stack of L matmuls
x <- tanh(x @ W_i), H x H fp16, run three ways for batch 1 and batch 4:

  const   weights embedded (rebel.compile_from_torch on a module with
          parameters), one compile per batch: each graph brings its weights;
  host    weights as graph inputs, passed from the host every run
          (compile_from_torch + Runtime.run with numpy arrays);
  device  weights as graph inputs that are torch-rbln device tensors
          (torch.compile(backend="rbln")); two graphs compiled separately run
          against the same device tensors.

Records device memory (rebel.get_device_memory_info when it answers, else
rbln-stat), per-call time and the max error against CPU.

  python bench/swap_atom_probe.py --out probe.json [--L 16 --H 4096]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch


def mem_used(device):
    """Device memory in use, bytes, by the documented utility or rbln-stat."""
    try:
        import rebel
        info = rebel.get_device_memory_info(device)
        for k in ("used", "used_memory", "allocated"):
            if hasattr(info, k):
                return int(getattr(info, k)), "rebel"
        return str(info), "rebel-raw"
    except Exception as e:  # noqa: BLE001 - fall back to the CLI
        err = repr(e)
    try:
        out = subprocess.run(["rbln-stat", "-j"], capture_output=True, text=True, timeout=30).stdout
        return out[:4000], "rbln-stat " + err
    except Exception as e2:  # noqa: BLE001
        return None, err + " / " + repr(e2)


def timed(fn, n=10):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


class Stack(torch.nn.Module):
    def __init__(self, ws):
        super().__init__()
        self.ws = torch.nn.ParameterList([torch.nn.Parameter(w, requires_grad=False) for w in ws])

    def forward(self, x):
        for w in self.ws:
            x = torch.tanh((x.to(w.dtype) @ w).float())
        return x


class StackInputs(torch.nn.Module):
    def forward(self, x, *ws):
        # fp32 activations, fp16 weights, as rngd-pd's ATOM decode does
        for w in ws:
            x = torch.tanh((x.to(w.dtype) @ w).float())
        return x


def ref(x, ws):
    x = x.float()
    for w in ws:
        x = torch.tanh((x.to(w.dtype).float() @ w.float()))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--modes", default="const,host,device")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import rebel
    torch.manual_seed(0)
    dt = torch.float16
    ws = [(torch.randn(a.H, a.H) / a.H ** 0.5).to(dt) for _ in range(a.L)]
    xs = {b: torch.randn(b, a.H) for b in (1, 4)}
    res = dict(L=a.L, H=a.H, weight_bytes=a.L * a.H * a.H * 2, rebel=rebel.__version__,
               mem_start=mem_used(a.device))
    modes = a.modes.split(",")

    if "const" in modes:
        r = res["const"] = {}
        rts = {}
        for b in (1, 4):
            t0 = time.time()
            cm = rebel.compile_from_torch(Stack(ws).eval(), example_inputs=[xs[b]],
                                          npu=os.environ.get("RBLN_PART", "RBLN-CA22"))
            r[f"compile_B{b}_s"] = time.time() - t0
            r[f"alloc_B{b}"] = cm.get_total_device_alloc()
            rts[b] = cm.create_runtime(device=a.device, tensor_type="pt")
            r[f"mem_after_B{b}"] = mem_used(a.device)
            r[f"step_B{b}_s"] = timed(lambda: rts[b].run(xs[b]))
            out = rts[b].run(xs[b])
            out = out[0] if isinstance(out, (list, tuple)) else out
            r[f"err_B{b}"] = float((out.float() - ref(xs[b], ws)).abs().max())
        del rts

    if "host" in modes:
        r = res["host"] = {}
        rts = {}
        for b in (1, 4):
            t0 = time.time()
            cm = rebel.compile_from_torch(StackInputs().eval(), example_inputs=[xs[b], *ws],
                                          npu=os.environ.get("RBLN_PART", "RBLN-CA22"))
            r[f"compile_B{b}_s"] = time.time() - t0
            r[f"alloc_B{b}"] = cm.get_total_device_alloc()
            rts[b] = cm.create_runtime(device=a.device, tensor_type="pt")
            r[f"mem_after_B{b}"] = mem_used(a.device)
            r[f"step_B{b}_s"] = timed(lambda: rts[b].run(xs[b], *ws))
            out = rts[b].run(xs[b], *ws)
            out = out[0] if isinstance(out, (list, tuple)) else out
            r[f"err_B{b}"] = float((out.float() - ref(xs[b], ws)).abs().max())
        del rts

    if "device" in modes:
        import torch_rbln  # noqa: F401  registers the rbln device
        r = res["device"] = {}
        dev = f"rbln:{a.device}"
        t0 = time.time()
        wd = [w.to(dev) for w in ws]
        r["upload_weights_s"] = time.time() - t0
        r["mem_after_weights"] = mem_used(a.device)
        for b in (1, 4):
            # a new torch.compile per batch: its own compile, as a separately
            # built variant would be
            g = torch.compile(StackInputs().eval(), backend="rbln", dynamic=False)
            xd = xs[b].to(dev)
            t0 = time.time()
            out = g(xd, *wd)
            r[f"compile_and_first_B{b}_s"] = time.time() - t0
            r[f"mem_after_B{b}"] = mem_used(a.device)
            r[f"step_B{b}_s"] = timed(lambda: g(xd, *wd).cpu())
            r[f"err_B{b}"] = float((out.cpu().float() - ref(xs[b], ws)).abs().max())
            r[f"out_device_B{b}"] = str(out.device)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
