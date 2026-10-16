#!/usr/bin/env python3
"""Compile and save ATOM decode artifacts without opening a card.

The vendor module's own --build path compiles, saves, and then creates a
runtime to time a step. That last part needs a free card, and on this host it
fails with "Failed to retrieve information for the requested NPU device (0)"
even when rbln-stat reports every card idle -- the device index the runtime
takes is not the index the tool prints. The exception then kills the loop, so
one unusable card costs every artifact after it in the list even though the
compile and the save both succeeded.

Baking is a host-side job and has no reason to touch a card at all, so this
does the two steps that matter and stops. Correctness is checked later by the
pod that loads the artifact, against the same CPU reference as before.

    python3 bench/rbln_bake.py --S 512 --batches 4 --suffix _lm
"""
import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--batches", default="4")
    ap.add_argument("--suffix", default="_lm")
    ap.add_argument("--artifact-dir", default="/home/user/rbln-llama8b-pd")
    a = ap.parse_args()

    import rebel
    from rngd_pd.vendors.rbln_decode_batch import (LlamaDecodeBatchRBLN,
                                                   example_inputs, RBLN_PART)
    from rngd_pd.vendors.rbln_decode import load_hf
    _tk, cfg, sd = load_hf()
    nl = cfg.n_layers
    for B in [int(b) for b in a.batches.split(",")]:
        art = f"{a.artifact_dir}/decode_S{a.S}_B{B}{a.suffix}.rbln"
        if os.path.exists(art):
            print(f"[skip] {art} exists", flush=True)
            continue
        mod = LlamaDecodeBatchRBLN(cfg, sd, a.S, nl, B).eval()
        t0 = time.time()
        # Naming the part keeps the compiler off the machine's device list,
        # which is what lets this run while the fleet holds every card.
        compiled = rebel.compile_from_torch(
            mod, example_inputs=example_inputs(cfg, a.S, nl, B), npu=RBLN_PART)
        print(f"[bake] B={B} S={a.S} compiled in {time.time()-t0:.0f}s", flush=True)
        compiled.save(art)
        print(f"[bake] saved -> {art} "
              f"({os.path.getsize(art)/1e9:.2f} GB)", flush=True)
        del compiled, mod


if __name__ == "__main__":
    main()
