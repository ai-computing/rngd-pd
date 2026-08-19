"""Put src/ on the path so a bench script can be run directly.

The package proper never needs this — inside rngd_pd, imports are ordinary
absolute imports. This exists only so `python3 bench/sweep_llmd.py` works
without setting PYTHONPATH first.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
