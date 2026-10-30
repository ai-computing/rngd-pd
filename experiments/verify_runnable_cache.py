"""Allocated-device cache roundtrip with alias, passthrough and empty outputs.

Run compile and load in separate processes with the same --cache and SDK.
The load phase forbids compilation. Outputs must equal the CPU reference.
"""
import argparse
import hashlib
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["compile", "load"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", type=int, required=True)
    a = ap.parse_args()
    from rngd_pd.engines import artifact_cache as ac
    from rngd_pd.engines.fsdk import CompileModule, SDK, NEW, dev_str, open_device, sync
    if not NEW:
        raise RuntimeError("this probe validates the new Runnable cache path")
    open_device()
    key = ac.cache_key(probe="runnable-metadata-v1", shape=(2, 128))
    path = ac.path_for(a.cache, key)

    class Tiny(torch.nn.Module):
        def forward(self, x, y):
            z = x + y
            return z, z, x, x[:0]

    x = torch.arange(256, dtype=torch.float32).reshape(2, 128).to(torch.bfloat16)
    y = torch.ones_like(x)
    expected = Tiny()(x, y)

    def build():
        return CompileModule.from_module(Tiny(), (x, y))

    if a.phase == "load":
        def forbidden(*args, **kwargs):
            raise AssertionError("fresh load must not compile")
        CompileModule.from_module = forbidden
        if not ac.exists(path):
            raise FileNotFoundError(path)
    elif ac.exists(path):
        raise FileExistsError("compile phase requires a fresh owned cache")
    t0 = time.perf_counter()
    m, mode, _ = ac.get_or_compile(a.cache, key, build)
    elapsed = time.perf_counter() - t0
    assert ac.exists(path)
    dev = dev_str(a.device)
    m.to(dev)
    inputs = [t.to(dev) for t in (x, y)]
    out = m(*inputs)
    sync(a.device)
    assert len(out) == len(expected)
    checks = []
    for actual, reference in zip(out, expected):
        assert actual.device.type == "furiosa", "native output left the device"
        cpu = actual.cpu()
        assert torch.equal(cpu, reference)
        checks.append(dict(shape=list(cpu.shape), dtype=str(cpu.dtype),
                           sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()))
    print(json.dumps(dict(complete=True, phase=a.phase, mode=mode, sdk=SDK,
                          torch=str(torch.__version__), cache_s=elapsed,
                          cache_bytes=ac.size(path), outputs=checks,
                          cpu_fallback=False)))


if __name__ == "__main__":
    main()
