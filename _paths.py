"""Portable path bootstrap — adds every package subdir to sys.path from __file__.
Import this once at the top of any script; then flat imports (generic_runtime, rngd_kv_connector,
registry, …) resolve no matter where the package is copied."""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("runtime", "connector", "gateway", "serving", "bench", "tests"):
    _p = os.path.join(_root, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
