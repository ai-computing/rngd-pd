"""Where things live on disk, worked out once.

Before the restructure each module derived the artifact cache from its own
__file__ with a different number of `..` hops, so moving a file silently
pointed it at a directory that did not exist and the compile happened again.
One constant, imported everywhere, makes that impossible.
"""
import os
from pathlib import Path

# src/rngd_pd/paths.py -> src/rngd_pd -> src -> repo root
ROOT = Path(__file__).resolve().parents[2]

# Compiled EDF graphs, keyed by graph hash. Shared with the pods through a
# hostPath mount, which is why it sits in the tree rather than under /var.
ARTIFACT_CACHE = Path(os.environ.get("RNGD_ARTIFACT_CACHE", ROOT / "artifacts_cache"))

# KV handoff between a prefill worker and a decode worker. Both pods mount the
# node's /dev/shm, so this path has to agree across them.
SHM_ROOT = Path(os.environ.get("PD_SHM_ROOT", "/dev/shm/rngd_kv_cluster"))
