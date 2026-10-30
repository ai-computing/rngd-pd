# Copyright 2026 The onggi authors. SPDX-License-Identifier: Apache-2.0
"""Synthetic artifacts of the golden switch plans (same bytes as pkg/store/conformance_test.go).

    python3 golden_data.py OUT_DIR    writes OUT_DIR/<case>/{base,target}/<files>
"""
import hashlib
import os
import sys

CASES = ("identical", "inplace", "shifted", "rearranged")
SIZE = 2 << 20
BLOCK = 32 << 10


def data(seed, n):
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(("onggi-conformance:%d:%d" % (seed, i)).encode()).digest()
        i += 1
    return bytes(out[:n])


def config(batch):
    return ('{"batch": %d}\n' % batch).encode()


def case(name):
    """Returns (base files, target files) as {path: bytes}."""
    base = data(1, SIZE)
    b = {"model.bin": base, "config.json": config(2)}
    if name == "identical":
        t = dict(b)
    elif name == "inplace":
        m = bytearray(base)
        m[1 << 20:(1 << 20) + (40 << 10)] = data(2, 40 << 10)
        t = {"model.bin": bytes(m), "config.json": config(8)}
    elif name == "shifted":
        t = {"model.bin": data(3, 17952) + base, "config.json": config(2)}
    elif name == "rearranged":
        idx = sorted(range(SIZE // BLOCK), key=lambda i: hashlib.sha256(("perm:%d" % i).encode()).digest())
        t = {"model.bin": b"".join(base[i * BLOCK:(i + 1) * BLOCK] for i in idx), "config.json": config(2)}
    else:
        raise KeyError(name)
    return b, t


def write(root, files):
    for path, content in files.items():
        p = os.path.join(root, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(content)


if __name__ == "__main__":
    for name in CASES:
        base, target = case(name)
        write(os.path.join(sys.argv[1], name, "base"), base)
        write(os.path.join(sys.argv[1], name, "target"), target)
