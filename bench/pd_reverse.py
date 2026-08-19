#!/usr/bin/env python3
"""Reversed PD: Rebellions ATOM prefills, Furiosa RNGD decodes.

The forward split put prefill on RNGD because that is the faster card per
token — 14.8 ms against ATOM's 68.3 ms. But prefill does not have to go a token
at a time: ATOM compiles a chunked graph that puts C positions through one
weight read, which drops its prefill cost to ~2 ms/token, while RNGD's eager
path has no multi-position graph and stays at 14.8 ms. So the fast prefiller and
the fast decoder are the opposite of what the forward split assumed.

The KV crosses the same way it did forward — the connector is direction-blind.
Correctness is judged against the same engine running the whole request alone.

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_reverse.py --D 8 --chip 2
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

import _bootstrap  # noqa: F401

torch.set_grad_enabled(False)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "connector")))


def handoff(Kh, Vh, nl, root="/dev/shm/rngd_kv_reverse"):
    """Push KV through the vLLM connector contract and read it back."""
    import shutil
    from rngd_pd.connector import transport as kv_transport
    kv_transport.WIRE = "bf16"
    from rngd_pd.connector.transport import ShmTransport
    from rngd_pd.connector.rngd_kv import RngdKVConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    req = "reverse-0"
    t0 = time.perf_counter()
    pf = RngdKVConnector(KVConnectorRole.WORKER, req_id=req,
                         transport=ShmTransport(root), n_layers=nl)
    for i in range(nl):
        pf.save_kv_layer(f"layer.{i}", (Kh[i], Vh[i]))
    pf.wait_for_save()
    save_ms = (time.perf_counter() - t0) * 1e3
    dk = [np.zeros_like(Kh[i]) for i in range(nl)]
    dv = [np.zeros_like(Vh[i]) for i in range(nl)]
    dc = RngdKVConnector(KVConnectorRole.WORKER, req_id=req,
                         transport=ShmTransport(root), n_layers=nl)
    dc.register_kv_caches((dk, dv))
    dc.bind_connector_metadata(dc.build_connector_meta(None))
    t1 = time.perf_counter()
    dc.start_load_kv()
    for i in range(nl):
        dc.wait_for_layer_load(f"layer.{i}")
    load_ms = (time.perf_counter() - t1) * 1e3
    K2, V2 = np.stack(dk), np.stack(dv)
    exact = bool(np.array_equal(K2, Kh) and np.array_equal(V2, Vh))
    return K2, V2, save_ms, load_ms, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--atom-device", type=int, default=1)
    ap.add_argument("--artifact", default="/home/user/rbln-llama8b-pd/prefill_S256_C16.rbln")
    ap.add_argument("--prompt", default="The three primary colors are red, "
                    "blue and yellow. The capital of France is")
    a = ap.parse_args()

    import rebel  # noqa: F401 — fail fast
    from rngd_pd.vendors.rbln_decode import MODEL, load_hf
    from rngd_pd.vendors.rbln_prefill import LlamaPrefillRBLN, build_runtime, chunk_inputs
    from rngd_pd.kernels import decode_resident as LER
    from rngd_pd.engines.decode_resident import build_engine, EngineSession
    from rngd_pd.engines.model import rmsnorm
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(MODEL)
    prompt = tk.encode(a.prompt, add_special_tokens=False)
    P = (len(prompt) // a.C) * a.C or a.C
    prompt = prompt[:P]

    # ---------- reference: the whole request on RNGD alone ----------
    print(f"[rngd] loading engine (chip {a.chip}, cached EDF)...", flush=True)
    cfg, rt = build_engine(MODEL, a.S, 32, chip=a.chip)
    sess = EngineSession(rt, LER)
    nl = rt.nl
    Kp, Vp = sess.fresh_kv()
    t0 = time.perf_counter()
    for pos, t in enumerate(prompt[:-1]):
        _, Kp, Vp = sess.step(t, pos, Kp, Vp)
    nxt_ref, Kp, Vp = sess.step(prompt[-1], P - 1, Kp, Vp)
    rngd_prefill_ms = (time.perf_counter() - t0) * 1e3
    ref = [nxt_ref]
    cur, p = nxt_ref, P
    tsr = []
    for _ in range(a.D - 1):
        t1 = time.perf_counter()
        cur, Kp, Vp = sess.step(cur, p, Kp, Vp)
        tsr.append((time.perf_counter() - t1) * 1e3)
        ref.append(cur)
        p += 1
    rngd_tpot = float(np.median(tsr)) if tsr else 0.0
    print(f"[rngd] reference {ref}  prefill {rngd_prefill_ms:.0f} ms "
          f"({rngd_prefill_ms/P:.1f} ms/tok) · decode {rngd_tpot:.1f} ms/tok",
          flush=True)
    del Kp, Vp

    # ---------- ATOM: chunked prefill ----------
    _tk, cfg2, sd = load_hf()
    embed = sd["model.embed_tokens.weight"]
    lm = sd["lm_head.weight"]
    mod = None
    if not os.path.exists(a.artifact):
        mod = LlamaPrefillRBLN(cfg2, sd, a.S, nl, a.C).eval()
    del sd
    art = build_runtime(mod, cfg2, a.S, nl, a.C, f"prefill {nl}L C={a.C}",
                        artifact=a.artifact, device=a.atom_device)
    del mod
    K = np.zeros((nl, cfg2.n_kv, cfg2.head_dim, a.S), np.float32)
    V = np.zeros((nl, cfg2.n_kv, a.S, cfg2.head_dim), np.float32)
    tsc = []
    t0 = time.perf_counter()
    for base in range(0, P, a.C):
        x, cos, sin, mask = chunk_inputs(cfg2, a.S, a.C, base, embed,
                                         prompt[base:base + a.C])
        t1 = time.perf_counter()
        out = art.run(x, K, V, cos, sin, mask)
        tsc.append((time.perf_counter() - t1) * 1e3)
        hid = np.asarray(out[0])
        K[:, :, :, base:base + a.C] = np.asarray(out[1])
        V[:, :, base:base + a.C, :] = np.asarray(out[2])
    atom_prefill_ms = (time.perf_counter() - t0) * 1e3
    nxt = int(np.argmax(lm @ hid[-1]))
    print(f"[atom] chunked prefill {P} tok in {atom_prefill_ms:.0f} ms "
          f"({atom_prefill_ms/P:.2f} ms/tok, {len(tsc)} chunks) next={nxt}",
          flush=True)

    # ---------- handoff ----------
    K2, V2, save_ms, load_ms, exact = handoff(K, V, nl)
    print(f"[llm-d] save {save_ms:.0f} ms / load {load_ms:.0f} ms · "
          f"{K.nbytes*2/2/1e6:.1f} MB bf16 · bit-exact {exact}", flush=True)

    # ---------- RNGD: decode from the handed-off KV ----------
    Kd, Vd = sess.import_kv(K2, V2)
    outs = [nxt]
    cur, p = nxt, P
    tsd = []
    for _ in range(a.D - 1):
        t1 = time.perf_counter()
        cur, Kd, Vd = sess.step(cur, p, Kd, Vd)
        tsd.append((time.perf_counter() - t1) * 1e3)
        outs.append(cur)
        p += 1
    tpot = float(np.median(tsd)) if tsd else 0.0

    match = sum(x == y for x, y in zip(outs, ref))
    e2e = atom_prefill_ms + save_ms + load_ms + sum(tsd)
    print("\n" + "=" * 72)
    print("REVERSED PD — Rebellions ATOM prefill → Furiosa RNGD decode")
    print(f"  {MODEL}  {nl}L · P={P} · D={a.D} · S={a.S} · C={a.C}")
    print(f"  prefill : ATOM {atom_prefill_ms:7.0f} ms  ({atom_prefill_ms/P:5.2f} ms/tok)"
          f"   vs RNGD {rngd_prefill_ms:.0f} ms ({rngd_prefill_ms/P:.1f} ms/tok)"
          f"  → {rngd_prefill_ms/atom_prefill_ms:.1f}x faster")
    print(f"  handoff : {save_ms + load_ms:7.0f} ms  (bit-exact {exact})")
    print(f"  decode  : RNGD {tpot:5.1f} ms/tok")
    print(f"  e2e     : {e2e:7.0f} ms for {P}+{a.D} tokens")
    print(f"  reverse : {outs}")
    print(f"  RNGD    : {ref}")
    print(f"  text    : {tk.decode(outs)!r}")
    print(f"  argmax agreement {match}/{len(outs)} "
          f"{'PASS ✅' if match >= len(outs) - 1 else 'CHECK'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
