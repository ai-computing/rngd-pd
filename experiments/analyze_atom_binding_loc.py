#!/usr/bin/env python3
"""Count explicit ATOM binding/public runtime and optional constant-engine scopes.

No SDK imports. Counts include entire files, comments and dormant helpers.
The builder sources are included conservatively in full_public_atom_runtime;
their deferred benchmark imports are not a recursive dependency closure.
"""
import argparse
import hashlib
import json
from pathlib import Path


ADAPTER = (
    "src/rngd_pd/swap/atom_binding.py",
    "src/rngd_pd/swap/binding_cli.py",
)
PARSER_POLICY = (
    "src/rngd_pd/swap/delta.py",
    "src/rngd_pd/swap/variants.py",
)
PUBLIC_RUNTIME_SUPPORT = (
    "src/rngd_pd/__init__.py",
    "src/rngd_pd/swap/__init__.py",
    "src/rngd_pd/swap/atom.py",
    "src/rngd_pd/swap/atom_bundles.py",
    "src/rngd_pd/engines/__init__.py",
    "src/rngd_pd/engines/model.py",
)
GRAPH_BUILD_SOURCE = (
    "src/rngd_pd/vendors/__init__.py",
    "src/rngd_pd/vendors/rbln_decode_batch.py",
    "src/rngd_pd/vendors/rbln_decode.py",
)
EXPERIMENTAL_ADDITIONS = (
    "src/rngd_pd/swap/atom_constants.py",
    "src/rngd_pd/swap/atom_inputs.py",
    "src/rngd_pd/swap/rngd.py",
    "src/rngd_pd/swap/load.py",
)


def report(root):
    public = ADAPTER + PARSER_POLICY + PUBLIC_RUNTIME_SUPPORT + GRAPH_BUILD_SOURCE
    scopes = {
        "adapter_only": ADAPTER,
        "parser_and_policy": PARSER_POLICY,
        "public_runtime_load_support": PUBLIC_RUNTIME_SUPPORT,
        "graph_build_source": GRAPH_BUILD_SOURCE,
        "full_public_atom_runtime": public,
        "experimental_constant_runtime_additions": EXPERIMENTAL_ADDITIONS,
        "full_with_experimental_constant_runtime": public + EXPERIMENTAL_ADDITIONS,
    }
    files = {}
    for path in sorted({path for paths in scopes.values() for path in paths}):
        data = (root / path).read_bytes()
        lines = data.decode("utf-8").splitlines()
        files[path] = {"sha256": hashlib.sha256(data).hexdigest(),
                       "physical": len(lines), "nonblank": sum(bool(line.strip()) for line in lines)}
    totals = {}
    for name, paths in scopes.items():
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate file in scope " + name)
        identity = [(path, files[path]["sha256"]) for path in sorted(paths)]
        totals[name] = {"files": list(paths), "file_count": len(paths),
                        "physical": sum(files[path]["physical"] for path in paths),
                        "nonblank": sum(files[path]["nonblank"] for path in paths),
                        "source_sha256": hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()}
    return {
        "schema": "atom-binding-loc.v1",
        "method": "UTF-8 splitlines; nonblank uses str.strip; whole files including comments/docstrings and dormant helpers",
        "source_identity": "SHA-256 of raw file bytes; scope SHA-256 over sorted compact JSON [path, file SHA-256] pairs; exact working-tree snapshot",
        "excluded": "Vendor SDK/dependency implementations, tests, docs, caches, generated artifacts, native harness and this script",
        "scope_notes": {
            "public_runtime_load_support": "Actual package initialization plus AtomSession/Tables/model support; atom.py unconditionally imports atom_bundles, whose Optimum loader is not invoked",
            "graph_build_source": "Conservatively includes public publisher build_session graph implementation and its top-level model import; these files are not imported by precompiled AtomSession loading",
            "experimental_constant_runtime_additions": "Optional #28 ConstantWeightDecode/DeviceWeightDecode and direct Checkpoint/load_delta support. Checkpoint uses host parsing; deferred SwappableDecode Furiosa construction and unrelated CLI benchmark dependencies are outside scope",
            "full_with_experimental_constant_runtime": "Union of public scope and optional engine additions, not a claim that the experimental ABI is public or that every dormant helper dependency is counted recursively",
        },
        "scopes": totals,
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    document = json.dumps(report(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(document, encoding="utf-8")
    else:
        print(document, end="")


if __name__ == "__main__":
    main()
