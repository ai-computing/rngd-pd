"""Bake a decode graph into artifacts_cache from the host, so no pod compiles it.

RNGD_MAX_SHAPES is a real win — at group=2 it takes the step from 26.20 to
22.55 ms and the bandwidth from 532 to 619 GB/s, and it lifts the group to 8
for 17.62 ms and 792 GB/s. It also makes the pod die: both configs raise
UnsupportedOpError inside the container where they compile on the host in 78s
and 369s. The pod holds the whole state dict before it starts and lives under
a 96Gi cap, so it has far less headroom than a bare process.

Rather than fight that, take compiling off the pod's critical path entirely.
artifacts_cache is a hostPath both sides mount, and the key comes from the same
decode_graph_spec the engine uses, so a graph baked here is the one the worker
looks for. No card is touched: get_or_compile is called with device=None.

    RNGD_MAX_SHAPES=8 python bench/prebake.py --B 4 --S 256 --group 8
"""
import argparse
import os
import time

import torch

from rngd_pd.engines.decode_batch import (decode_graph_spec, lmhead_graph_spec,
                                          MAX_SHAPES)
from rngd_pd.engines.artifact_cache import get_or_compile, exists, path_for
from rngd_pd.paths import ARTIFACT_CACHE

# The .w half of an artifact is a torch.save pickle carrying a pytree treespec,
# and it does not survive a torch version change: a 2.10-baked graph loads in a
# 2.11 pod without error and then fails at call time with "Given arguments
# don't match expected argument spec". The pods put ~/.local ahead of the venv
# on PYTHONPATH, so they run furiosa-torch against torch 2.11 even though the
# venv ships 2.10. Bake with the same combination or the artifact is poison:
#   PYTHONPATH=src:/home/user/.local/lib/python3.10/site-packages \
#     /home/user/venvs/furiosa/bin/python bench/prebake.py ...
POD_TORCH = os.environ.get("POD_TORCH", "2.11")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--cache", default=str(ARTIFACT_CACHE))
    a = ap.parse_args()

    if not torch.__version__.startswith(POD_TORCH):
        raise SystemExit(
            f"torch {torch.__version__} 로는 구우면 안 된다 — 파드는 {POD_TORCH}.x 를 쓴다.\n"
            f"버전이 어긋난 아티팩트는 조용히 로드된 뒤 호출 시점에 터진다.\n"
            f"PYTHONPATH 에 /home/user/.local/lib/python3.10/site-packages 를 앞세워 다시 실행할 것.")
    print(f"torch {torch.__version__} (파드와 일치)")

    cap = MAX_SHAPES.get("max_num_unique_shapes", "무제한")
    print(f"B={a.B} S={a.S} group={a.group} shapes={cap}")
    os.makedirs(a.cache, exist_ok=True)

    # Both graphs the pod would otherwise compile. Baking only the decode one
    # left the lm_head behind, and at B=8 the pod dies compiling it -- the
    # traceback names the lm_head, so the batch width never looks like the
    # cause. The two belong in the same bake.
    from transformers import AutoConfig
    V = AutoConfig.from_pretrained(a.model).vocab_size
    for label, (key, build) in (
            ("decode", decode_graph_spec(a.B, a.S, a.group)),
            ("lmhead", lmhead_graph_spec(a.B, V))):
        target = path_for(a.cache, key)
        print(f"[{label}] 키 {key}")
        if exists(target):
            print(f"[{label}] 이미 구워져 있다 — 파드가 그대로 로드한다")
            continue
        t0 = time.time()
        _m, mode, secs = get_or_compile(a.cache, key, build, device=None)
        assert exists(target), f"[{label}] 컴파일은 됐는데 캐시에 안 남았다"
        size = sum(os.path.getsize(target + e) for e in (".edf", ".w"))
        print(f"[{label}] {mode} in {secs:.0f}s, 캐시에 기록됨 {size/1e6:.1f} MB "
              f"(총 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
