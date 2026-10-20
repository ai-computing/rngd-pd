#!/usr/bin/env python3
"""Fill the site placeholders in a manifest and print it.

The manifests here pin the workers to one node and mount Python trees from the
home directory of the account that owns them. Both are site values, so the
tracked YAML carries ${RNGD_PD_NODE} and ${RNGD_PD_HOME} instead. Set them in
cluster.local.env at the repo root (see cluster.example.env) or export them.

  python3 deploy/k8s/render.py deploy/k8s/pd-hetero.yaml | kubectl apply -f -
  python3 deploy/k8s/gen_sweep.py | python3 deploy/k8s/render.py - | kubectl apply -f -

Only these names are replaced; every other `$` in the manifests (shell in the
container commands) is left alone.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench"))
import cluster_env  # noqa: E402  (loads cluster.local.env)

SITE = ("RNGD_PD_NODE", "RNGD_PD_HOME")
PLACEHOLDER = re.compile(r"\$\{(RNGD_PD_[A-Z0-9_]+)\}")


def render(text, env=None):
    env = os.environ if env is None else env
    unknown = sorted({m for m in PLACEHOLDER.findall(text) if m not in SITE})
    if unknown:
        raise SystemExit(f"unknown placeholder(s): {', '.join(unknown)}")
    values = {}
    for name in SITE:
        if "${%s}" % name in text:
            values[name] = env.get(name) or cluster_env.require(name)
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def main(argv):
    if len(argv) != 1:
        raise SystemExit("usage: render.py MANIFEST.yaml | -")
    if argv[0] == "-":
        data = sys.stdin.buffer.read()
    else:
        with open(argv[0], "rb") as f:
            data = f.read()
    # bytes in, bytes out: UTF-8 and line endings pass through unchanged
    sys.stdout.buffer.write(render(data.decode("utf-8")).encode("utf-8"))


if __name__ == "__main__":
    main(sys.argv[1:])
