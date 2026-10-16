"""Site-specific endpoints for the bench scripts, read from the environment.

Gateway addresses and the way to reach kubectl differ per cluster, so they are
not in the code. Put them in `cluster.local.env` at the repo root (untracked;
copy `cluster.example.env`) or export them. A variable already set in the
environment wins over the file.
"""
import os
import shlex
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_ENV = os.path.join(_ROOT, "cluster.local.env")

# llm-d gateway route that serves the PD configurations, below LLMD_GATEWAY_URL
ROUTE = "/rngd/llama8b-pd/completions"


def _load():
    if not os.path.exists(LOCAL_ENV):
        return
    with open(LOCAL_ENV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            os.environ.setdefault(key, val.strip().strip("\"'"))


_load()


def require(name):
    val = os.environ.get(name)
    if not val:
        sys.exit(f"{name} is not set. Export it or add it to {LOCAL_ENV} "
                 f"(see cluster.example.env).")
    return val


def gateway():
    """Full completions URL. PD_GATEWAY (a full URL) overrides the default route."""
    if os.environ.get("PD_GATEWAY"):
        return os.environ["PD_GATEWAY"]
    return require("LLMD_GATEWAY_URL").rstrip("/") + ROUTE


def kubectl():
    """kubectl argv prefix; KUBECTL_CMD may wrap it, e.g. in ssh to the control node."""
    return shlex.split(os.environ.get("KUBECTL_CMD", "kubectl"))
