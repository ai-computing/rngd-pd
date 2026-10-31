#!/usr/bin/env python3
"""Count explicit RNGD binding/runtime source scopes without importing an SDK.

Run from any directory; paths are relative to this repository. Physical lines
include comments and docstrings; nonblank lines exclude only whitespace lines.
This is whole-file accounting, not a count of statements reached by a request.
"""
import argparse
import hashlib
import json
from pathlib import Path


ADAPTER = (
    "src/rngd_pd/swap/rngd_binding.py",
    "src/rngd_pd/swap/binding_cli.py",
)
PARSER_POLICY = (
    "src/rngd_pd/swap/delta.py",
    "src/rngd_pd/swap/variants.py",
)
INHERITED_RUNTIME = (
    "src/rngd_pd/__init__.py",
    "src/rngd_pd/_tv_patch.py",
    "src/rngd_pd/paths.py",
    "src/rngd_pd/swap/__init__.py",
    "src/rngd_pd/swap/load.py",
    "src/rngd_pd/swap/rngd.py",
    "src/rngd_pd/engines/__init__.py",
    "src/rngd_pd/engines/artifact_cache.py",
    "src/rngd_pd/engines/decode_batch.py",
    "src/rngd_pd/engines/fsdk.py",
    "src/rngd_pd/engines/model.py",
    "src/rngd_pd/kernels/__init__.py",
    "src/rngd_pd/kernels/decode_batch.py",
    "src/rngd_pd/kernels/dsl_batch.py",
)
SWAP_INVENTORY = (
    "src/rngd_pd/swap/__init__.py",
    "src/rngd_pd/swap/atom.py",
    "src/rngd_pd/swap/atom_bundles.py",
    "src/rngd_pd/swap/atom_constants.py",
    "src/rngd_pd/swap/atom_inputs.py",
    "src/rngd_pd/swap/bench_atom.py",
    "src/rngd_pd/swap/bench_atom_bundles.py",
    "src/rngd_pd/swap/bench_rngd.py",
    "src/rngd_pd/swap/binding_cli.py",
    "src/rngd_pd/swap/delta.py",
    "src/rngd_pd/swap/load.py",
    "src/rngd_pd/swap/rngd.py",
    "src/rngd_pd/swap/rngd_binding.py",
    "src/rngd_pd/swap/variants.py",
)


def report(root):
    scopes = {
        "adapter_only": ADAPTER,
        "parser_and_policy": PARSER_POLICY,
        "inherited_rngd_runtime": INHERITED_RUNTIME,
        "full_rngd_runtime": ADAPTER + PARSER_POLICY + INHERITED_RUNTIME,
        "swap_directory_inventory": SWAP_INVENTORY,
    }
    files = {}
    for path in sorted(set(path for paths in scopes.values() for path in paths)):
        data = (root / path).read_bytes()
        lines = data.decode("utf-8").splitlines()
        files[path] = {"sha256": hashlib.sha256(data).hexdigest(),
                       "physical": len(lines), "nonblank": sum(bool(line.strip()) for line in lines)}
    result = {}
    for name, paths in scopes.items():
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate file in scope " + name)
        identities = [(path, files[path]["sha256"]) for path in sorted(paths)]
        result[name] = {"files": list(paths), "file_count": len(paths),
                        "physical": sum(files[path]["physical"] for path in paths),
                        "nonblank": sum(files[path]["nonblank"] for path in paths),
                        "source_sha256": hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()}
    return {"schema": "rngd-binding-loc.v1", "method": "UTF-8 splitlines; nonblank uses str.strip; comments/docstrings included",
            "excluded": "SDK/dependency implementations, tests, docs, caches, generated artifacts and this script",
            "scopes": result, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write JSON here; otherwise print JSON")
    args = parser.parse_args()
    document = json.dumps(report(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(document, encoding="utf-8")
    else:
        print(document, end="")


if __name__ == "__main__":
    main()
