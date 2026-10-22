# Delta load: swap compiled variants on a running engine

Issue: [#17](https://github.com/ai-computing/rngd-pd/issues/17). Code: [`src/rngd_pd/swap/`](../src/rngd_pd/swap).

A compiled variant is one shape of the same Llama-3.1-8B graph: a batch width `B` and a sequence bucket `S`.
Switching variants used to mean a full model load. Delta load keeps on the card what is already there and
loads only what differs.

## RNGD

rngd-pd's batched decode graph ([`engines/decode_batch.py`](../src/rngd_pd/engines/decode_batch.py)) takes every
weight as a graph input. The compiled EDF holds only the program, so its artifact is 1–6 MB, not 16 GB.

[`SwappableDecode`](../src/rngd_pd/swap/rngd.py) keeps **one** device copy of the weights in that layout:
- the projections, stacked per layer group;
- the chunked LM head.

Every variant of the same layer group runs against this copy.

`swap(variant)` does only the per-variant work:
1. Loads that variant's decode and LM-head EDFs from the artifact cache.
2. Builds the `[B,H]` norm rows and the position tables of `S`.
3. Makes the variant active.

The KV cache of the new shape is created for the next request. A variant that has been prepared once stays loaded,
so swapping back to it does nothing.

```python
from rngd_pd.swap.rngd import SwappableDecode
from rngd_pd.swap.variants import Variant, BatchSelector

eng = SwappableDecode("<Llama-3.1-8B-Instruct dir>", group=8, chip=0)   # weights uploaded once
eng.swap(Variant(4, 512))
outs, _ = eng.generate(prompts_4, n_new=64)

sel = BatchSelector([4, 8, 16], start=4)      # queue depth -> batch, with hysteresis
eng.swap(Variant(sel.observe(queue_depth), 512))
```

The graphs must already be in the artifact cache. `swap()` refuses to compile on the serving path. Bake them on
the host with `RNGD_MAX_SHAPES=8 bench/prebake.py --B <B> --S <S> --group 8`.

### Partial weight update

When the weights themselves change in part, `apply_delta(new_checkpoint, plan)` uploads only the tensors whose bytes
changed. A new checkpoint might share most bytes with the resident one, for example a fine-tune of a few layers.

The plan comes from [`swap/delta.py`](../src/rngd_pd/swap/delta.py):
1. `onggi diff --ranges --json` lists the changed byte ranges of each file.
2. The safetensors header, an open format, maps each range to the tensors and rows it overlaps.
3. onggi's ranges are chunk-granular, so a chunk that straddles a tensor boundary also flags its neighbour.
   `refine_rows` reads just the flagged rows from both files and keeps the rows that really differ.

Without onggi, the files are compared block by block.

```python
from rngd_pd.swap import delta
plan = delta.plan_update(base_dir, new_dir, delta.load_onggi_ranges("ranges.json"))
eng.apply_delta(new_dir, plan)          # mode="inplace" (per layer) or "group" (re-stack)
```

### Measured

Measured on one RNGD card (furiosa-torch 2026.2.0) in a Kubernetes Job. The bench is
`python -m rngd_pd.swap.bench_rngd all`, from [`deploy/k8s/swap-rngd-job.yaml`](../deploy/k8s/swap-rngd-job.yaml).

Setup:
- **Model:** Llama-3.1-8B-Instruct.
- **Graphs:** decode graphs at group 8 with `RNGD_MAX_SHAPES=8`, from the artifact cache.
- **Prompts:** 16 prompt tokens fed through the decode graph, then 16 greedy tokens per stream.
- **Repeats:** two full runs.

Definitions:
- **Full reload:** a fresh process, from launch to the first token: read the checkpoint, upload the weights,
  load the graphs, first step.
- **Swap:** in the running engine, from `swap()` to the first token on the new variant.

| Variant | Full reload (s) | Swap, first time (ms) | Swap back, already prepared (ms) | Greedy tokens = full reload |
|---|--:|--:|--:|:-:|
| B4, S512 | 21.8 / 22.3 | (initial load) | 113 / 145 | yes |
| B8, S512 | 22.0 / 22.2 | 728 / 941 | 148 / 275 | yes |
| B16, S512 | 22.7 / 22.5 | 867 / 1197 | 353 / 539 | yes |
| B4, S256 | 21.4 / 22.3 | 547 / 732 | 56 / 63 | yes |

Notes:
- Each cell gives run 1 / run 2. Tokens were compared for every stream, in both runs.
- **Where the swap time goes:** reading the two EDFs is 0.5–0.7 s; the rest is the first step on the new shape.
  Swapping back is the first step only.
- **Full reload:** 19–20 s of it is the 15 GB weight upload. The rest is the graph load (0.6–0.7 s) and the first
  step.
- **Device memory:**
  - free capacity is 33.5 GiB after the weights, on a 47.25 GiB card;
  - it stays 33.5 GiB with the first three variants loaded, and 33.25 GiB with all four;
  - furiosa-smi does not show furiosa-torch allocations, so this is measured by allocating 256 MiB chunks. Each
    variant therefore costs less than 0.25 GiB, against about 14 GiB for a second weight copy.

Partial weight update:
- **New checkpoint:** three tensors of the base were changed in place (`layers.30.q_proj`, `layers.31.down_proj`,
  `layers.31.post_attention_layernorm`), with the same file layout.
- **onggi:** `onggi diff --ranges` over the two changed shards gave 97.5 % same offset and two changed runs
  (151 MB).

| | Run 1 (onggi ranges as is) | Run 2 (ranges refined) |
|---|--:|--:|
| Tensors uploaded | 8 (5 flagged only by chunk boundaries, `lm_head` among them) | 3 |
| Bytes uploaded | 1.24 GB | 151 MB |
| Planning / upload (s) | 0.002 / 0.61 | 0.17 / 0.24 |
| Request → first token after the update (s) | 1.21 | 0.84 |
| Full reload of the new checkpoint (s) | 19.8 | 22.1 |
| Greedy tokens = full reload of the new checkpoint | yes | yes (and different from the base's) |

### Limits

- Variants share weights only within one layer group: the weights are stacked by group. A group-2 graph needs its
  own copy.
- `swap()` happens between requests. A request's KV cache is not carried into a variant of another shape.
- The weights live in the process that owns the device tensors. Another process cannot attach to them; furiosa-torch
  has no public API to share device tensors across processes.

ATOM: see [#17](https://github.com/ai-computing/rngd-pd/issues/17) (in progress).
