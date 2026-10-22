# Delta loading and graph swapping

`rngd_pd.swap` adds an opt-in engine API for selecting a compiled shape while
retaining device weights. It does not change the default cluster worker or
transparently migrate an in-flight request. Drain requests before selecting a
different shape or checkpoint; start new requests with fresh KV. Locks serialize
generation and swaps. RNGD rejects KV created before a shape or weight change.

## RNGD: shared device inputs

The batched decode graph accepts weights as inputs. `SwappableDecode` holds one
device copy of the projections and LM head and supplies it to each variant.
Only programs, position tables and batch-shaped normalization tensors belong to
a variant. Compilation is disabled on the swap path by default.

```python
from rngd_pd.swap import Variant, BatchSelector
from rngd_pd.swap.rngd import SwappableDecode

small, wide = Variant(4, 256, 8), Variant(8, 256, 8)
engine = SwappableDecode(checkpoint_dir, chip=0, group=8, cache_dir=artifact_cache)
engine.swap(small)
engine.prepare(wide)                  # prepare before traffic needs it
engine.swap(wide)                     # retains the projection and LM-head tensors
tokens, ttft = engine.generate([prompt_ids] * 8, n_new=8)

policy = BatchSelector([4, 8], start=8)
engine.select(policy, depth=3, context=128, variants=[small, wide])
engine.close()
```

Use the same `RNGD_MAX_SHAPES`, compiler settings, SDK **and PyTorch version**
when baking and loading the artifact cache. The existing `.w` files contain
pickled PyTorch tree specifications; an SDK-version directory alone does not
make them portable across PyTorch versions. `bench/prebake.py` documents the
environment used for the existing serving artifacts. Graph-spec lookup is now
lazy: a cache hit does not allocate compiler sample tensors.

The implementation currently requires the Llama-3.1-8B dimensions
(4096/14336, 32 attention heads, 8 KV heads, head dimension 128). Checkpoint
configuration is checked before an update; a different architecture, rotary
configuration, tensor set, shape or dtype requires a full reload. The existing
EDF artifact-cache path is the tested path; newer SDK cache formats require
their own compatible loader.

## Updating only changed RNGD weights

```python
from rngd_pd.swap.delta import load_onggi_ranges, plan_update

ranges = load_onggi_ranges("ranges.json")  # onggi diff --ranges --json
plan = plan_update(base_checkpoint_dir, new_checkpoint_dir, ranges)
result = engine.apply_delta(new_checkpoint_dir, plan, mode="inplace")
```

The planner reads the open safetensors headers and maps changed file spans to
tensor names. The range input must describe the **checkpoint files**, not the
compiled `.rbln`, `.edf` or `.fxb` files. A compiled-artifact byte offset is not
a checkpoint tensor offset or a device address. File moves and changed tensor
layouts are treated conservatively. Removed tensors and incompatible graph
configuration are rejected. Keep both checkpoint directories immutable while a
plan is in use.

`inplace` uploads complete changed projection tensors into their packed layer
slots. It validates and stages updates before writing and restores old tensor
values if a device write fails. Serving is disabled if rollback fails. `group`
stages replacement group stacks and commits their references together; it uses
more temporary HBM and uploads unchanged neighbours in each affected group.
Embedding lookup stays on the host. Tiled normalization inputs are rebuilt when
their weights change. Updates invalidate old KV even if the shape is unchanged.

Without a range input the fallback compares checkpoint files in blocks, reading
both full checkpoints. Its planning time and coarse extra tensor uploads must
be included in latency. Precomputed ranges avoid that scan. A diff with changed
runs but without `--ranges` is rejected.

## ATOM: shared compilation sessions and prepared bundles

Compile a bundle with all needed decoder batch sizes in one weight-sharing
session, for example using optimum-rbln's `rbln_decoder_batch_sizes=[1, 4, 8]`
with `rbln_batch_size=8`. Save the entire bundle, including its prefill weight
owner. Loading a decoder file alone is insufficient.

```python
from rngd_pd.swap.atom import AtomBundle, SwappableAtom

v1, v8 = Variant(1, 512), Variant(8, 512)
engine = SwappableAtom({"main": AtomBundle(compiled_bundle_dir, (v1, v8))}, device=0)
engine.swap(v1, "main")              # initial full bundle load
engine.swap(v8, "main")              # selects the co-compiled decoder
tokens, ttft = engine.generate([prompt_ids] * 8, 8, timed=True)
engine.close()
```

The model and all its runtimes stay alive. `run(*inputs)` exposes the selected
raw decoder runtime; `generate()` drives greedy generation with the matching
batch width. `select()` provides the same queue-depth policy hook as RNGD.

For an independently compiled bundle, register another `AtomBundle` and call
`future = engine.prepare_async(variant, "other")`. Current requests keep running
while one background thread loads it. After `future.result()`,
`engine.swap(variant, "other", require_ready=True)` selects it without loading
weights on that call. A failed preparation leaves the active bundle usable.
`release("main")` releases an inactive bundle; `allocations()` reports the SDK's
per-graph allocations for each loaded bundle.

This is a full load during preparation, with an additional weight copy for an
independent compilation session. Two 8B bundles may not fit a 16 GiB card.
Same-session switching shares weights; independent-session warm preparation
does not establish arbitrary partial ATOM weight replacement.

## Reproduce latency and correctness

```bash
PYTHONPATH=src python -m pytest -q tests/swap

RNGD_MAX_SHAPES=8 python -m rngd_pd.swap.bench_rngd all \
  --ckpt "$CHECKPOINT" --group 8 --chip 0 --cache "$CACHE" \
  --variants B4,S256 B8,S256 B4,S512 --P 8 --n 8 --out rngd.json

python -m rngd_pd.swap.bench_atom --bundle "$ATOM_BUNDLE" \
  --variants B1,S512 B8,S512 --tokens 8 --rounds 3 --out atom.json
```

The harnesses compare token IDs with fresh loads and exit nonzero on a mismatch.
They distinguish program selection, preparation and request-to-first-token
latency. ATOM uses a generation streamer to timestamp the first generated token;
RNGD timestamps the first prompt result before continuing generation. Raw
artifact paths and arguments belong in local run records, not public evidence.
See the experiment record for measured hardware, versions and limitations.
