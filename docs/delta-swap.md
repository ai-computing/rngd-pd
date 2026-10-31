# Delta loading and graph swapping

`rngd_pd.swap` adds an opt-in engine API for selecting a compiled shape while
retaining device weights. It does not change the default cluster worker or
transparently migrate an in-flight request. Drain requests before selecting a
different shape or checkpoint; start new requests with fresh KV. The RNGD,
ATOM device-input and optimum-bundle wrappers lock generation and swaps.
RNGD and ATOM device-input engines reject KV created
before a shape or weight change.

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

SDK 2026.4 uses a different cache loader. Native Runnable bytes and Torch
weights/output reconstruction metadata are published together in one atomic
`.runnable` archive under `CACHE/SDK/TORCH/KEY`. The loader checks all format
and version metadata before restoring the trusted executable/pickle payloads.
Alias, passthrough and empty output metadata must survive a fresh-process load.
SDK 2026.2 continues to use its existing `CACHE/SDK/KEY.edf` and `.w` pair.
Do not rename/copy old EDF entries into a new SDK directory.

For the installed SDK 2026.4 / Torch 2.10 CPU environment, the validated
B4/S256 graph uses **group=2**. Group=8 was rejected by the native compiler
with a memory-explosion error even with a 128 GiB container. Bake with the same
interpreter that the worker uses and explicitly select its Torch version:

```bash
POD_TORCH=2.10 RNGD_MAX_SHAPES=8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python bench/prebake.py --model "$CHECKPOINT" --B 4 --S 256 --group 2 --cache "$CACHE"
```

Use `SwappableDecode(..., group=2)` and `Variant(4, 256, 2)` for that setting;
the constructor default remains group=8. Other shapes need separate compilation
and validation. See the [native 2026.4 record](../experiments/2026-10-30-rngd-sdk264.md).
New archives honor the builder's umask; replacement preserves existing file
permissions so shared readers work without making private entries public.

Initial host weight layout can use two workers with
`SwappableDecode(..., host_workers=2)`; default 1 preserves serial CPU usage.
Checkpoint reads and device operations stay on the calling thread. Only pure
CPU transforms run in the bounded pool, which is joined before startup returns
or raises. Two workers at Torch intra-op=4 may use eight CPU threads. Prepared
worker configs accept the same RNGD-only option; it does not accelerate device
DMA itself or change resident row-update arithmetic.

This is a custom CompileModule cache, separate from Furiosa-LLM's FXB model
loader. It does not add live FXB weight replacement. The SDK wrapper exposes
its native serializer through `_inner`; that small adapter boundary must be
revalidated with SDK upgrades. Cache files must come from trusted builds.

The implementation currently requires the Llama-3.1-8B dimensions
(4096/14336, 32 attention heads, 8 KV heads, head dimension 128). Checkpoint
configuration is checked before an update; a different architecture, rotary
configuration, tensor set, shape or dtype requires a full reload. The existing
EDF path and the SDK 2026.4 Runnable path use their respective compatible
loaders; other SDK/cache combinations require independent validation.

## onggi to NPU in one call

Both `SwappableDecode` (RNGD) and `DeviceWeightDecode` (ATOM) expose the same
checkpoint update entry point:

```python
result = engine.load_delta(
    new_checkpoint_dir, store=onggi_store, base_ref="base", new_ref="new",
    onggi="/path/to/onggi",
)
tokens, ttft = engine.generate(prompts, n_new=8)
```

This runs `onggi diff --store STORE --ranges --json base new`, maps its output
to checkpoint tensors, refines chunk boundaries, and updates the resident NPU
engine. The refs must describe the matching immutable safetensors checkpoint
directories. Publish them before serving with `onggi push --store STORE --ref
base BASE_DIR` and the corresponding `new` command. File delivery/pulling and
the initial store upload are separate from the NPU update.

`result` reports `planning_s` (including the live onggi invocation), `delta_s`,
`total_s`, `uploaded_bytes` (weight payload), and the tensor plan. It does not
include the following prompt's first-token latency. `timeout` bounds the onggi
subprocess. A failed onggi command fails the update; it never silently falls
back to a full scan. Every target safetensors file must be represented in the
diff, so compiled-artifact diffs or incomplete manifests cannot become a
silent no-op.

For a precomputed diff, use `engine.load_delta(new_dir, ranges="ranges.json")`.
With no ranges/store options, it compares the two checkpoint directories
locally. Checkpoints and refs must remain immutable through the operation.

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

`inplace` uses the plan's changed checkpoint rows for projections and the LM
head. Projection rows become columns of the transposed resident layer; only
those rows are converted and staged, including the old values needed for
rollback. Contiguous full-layer and LM-row writes copy directly from the CPU;
sparse projection columns use a small temporary device patch, the faster route
in the validated native probes. The original projection/LM allocations remain
in use. LM runs split at chunk boundaries and leave vocabulary padding alone.

Missing row metadata in a manually constructed plan retains whole-tensor
behavior; changed header placement conservatively updates the complete tensor.
After coalescing, more than four separate runs or at least half the tensor's
rows also selects a whole-tensor update, bounding native dispatch overhead.
The engine validates row bounds, layouts and graph configuration before writing.
Failed or interrupted updates restore prior values and references. Serving is
disabled if rollback fails or is interrupted. `group`
stages replacement group stacks and commits their references together; it uses
more temporary HBM and uploads unchanged neighbours in each affected group.
Embedding lookup stays on the host. Tiled normalization inputs are rebuilt when
their weights change. Old KV is invalidated before a potentially mutating
transaction, including one whose writes are subsequently restored.

`uploaded_bytes` is the logical prepared weight payload, not a hardware PCIe
counter. The legacy `plan.upload_bytes`/summary field still counts full changed
checkpoint tensors, so it differs from sparse row payloads. Update results also
report `validation_s`, `stage_s`, `rollback_stage_s` and `copy_s`; staging can
include replacement allocations in `group` mode. `total_s` from `load_delta`
includes live onggi/planning and the complete update. Measure the next prompt's
first token separately to include any deferred device work.

Without a range input the fallback compares checkpoint files in blocks, reading
both full checkpoints. The integrated planner refines flagged rows by default,
retaining the behavior merged in PR #21; `refine=False` keeps the coarse plan.
Its planning time and any extra tensor uploads must
be included in latency. Precomputed ranges avoid that scan. A diff with changed
runs but without `--ranges` is rejected.

## ATOM: shared compilation sessions and prepared bundles

Compile a bundle with all needed decoder batch sizes in one weight-sharing
session, for example using optimum-rbln's `rbln_decoder_batch_sizes=[1, 4, 8]`
with `rbln_batch_size=8`. Save the entire bundle, including its prefill weight
owner. Loading a decoder file alone is insufficient.

```python
from rngd_pd.swap.atom import AtomBundle, SwappableAtom

v1, v8 = Variant(1, 512, group=0), Variant(8, 512, group=0)
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
`future = engine.prepare_async(variant, "other")`. Loading runs in a background
thread without taking the request lock. The vendor runtime can still serialize
native loading and execution; this does not guarantee uninterrupted latency.
After `future.result()`,
`engine.swap(variant, "other", require_ready=True)` selects it without loading
weights on that call. A failed preparation leaves the active bundle usable.
`release("main")` releases an inactive bundle; `allocations()` reports the SDK's
per-graph allocations for each loaded bundle.

This is a full load during preparation, with an additional weight copy for an
independent compilation session. Two 8B bundles may not fit a 16 GiB card.
Same-session switching shares weights; independent-session warm preparation
does not establish arbitrary partial ATOM weight replacement.

The deployed cluster worker can now use the native `AtomSession` or RNGD engines
directly through [prepared serving](prepared-serving.md). It selects prepared
B/S shapes only between drained batches and checks prefill checkpoint identity.

## ATOM: partial weights through device inputs

```python
from rngd_pd.swap.atom_inputs import DeviceWeightDecode

engine = DeviceWeightDecode(base_checkpoint_dir, cache_dir=graph_cache, device=0)
engine.swap(Variant(1, 64, group=0))
engine.generate([prompt_ids], n_new=8)  # initial compile and device preparation
result = engine.load_delta(new_checkpoint_dir, store=onggi_store,
                           base_ref="base", new_ref="new")
tokens, ttft = engine.generate([prompt_ids], n_new=8)
engine.close()
```

This is an actual partial checkpoint update on ATOM. The decode graph takes
FP16 projections/LM head and FP32 normalization weights as device inputs.
Changed tensors get new device allocations; unchanged allocations and the
compiled graph remain in use. The embedding stays on the host and is replaced
only when it changes. The update validates graph configuration, tensor sets,
shapes, dtypes and plan provenance before changing active state.

torch-rbln may defer physical allocation/layout conversion until a tensor is
used. The update therefore runs one scratch step with the staged inputs and
synchronizes before committing them. Failed staging/probing retains the old
input references and checkpoint. A fatal vendor/device failure can still
require an engine restart. The probe time is included in `delta_s` and is
also reported as `materialize_s`. Extra device headroom is needed for the
changed tensors and scratch KV; it does not require a second full model.

This opt-in path uses rebel 0.11.0 and torch-rbln 0.2.2. Initial graphs must be
compiled in the serving process on that stack; another process's device-input
cache can abort execution. Weight updates of the same shapes reuse the
already compiled graph. It retains the upstream device-input graph's slower
decode rate; see [the measured tradeoffs](delta-load.md). Embedded-weight
`.rbln`/optimum bundles use the session/bundle APIs above and require a reload
for changed weights. A byte-range diff of two such opaque compiled files is
not a device weight map.

## Reproduce latency and correctness

### Experimental ATOM constant-pool refresh

`rngd_pd.swap.atom_constants.ConstantWeightDecode` adds a separate fast-constant
path on the validated rebel 0.11.0 stack. Launch the dedicated worker with
`RBLN_WEIGHT_FREE=1` **before importing rebel**. Compile shapes in that process;
existing ordinary `.rbln` bundles lack the required meta-load mapping.

```python
from rngd_pd.swap.atom_constants import ConstantWeightDecode

engine = ConstantWeightDecode(base_checkpoint_dir, cache_dir=private_cache, device=0)
small = Variant(1, 64, 0)
engine.prepare(small)                 # compilation/materialization before admission
engine.swap(small)
tokens, ttft = engine.generate([prompt_ids], n_new=8)
result = engine.load_delta(new_checkpoint_dir, store=onggi_store,
                           base_ref="base", new_ref="new")
engine.close()
```

The updater maps checkpoint tensors to retained CPU buffers and their compiled
meta-load keys. It stages complete affected transform dependency groups,
resets the native already-applied bookkeeping, and writes those groups into
the existing runtime's constants. Original parameter identities and compiled
programs remain alive. Failed writes restore old groups; failed or interrupted
rollback disables the engine. Updates are serialized and invalidate old KV.

`changed_weight_bytes` counts changed converted tensors; `transform_input_bytes`
also includes companion tensors required by a complete transform group. The
compatibility field `uploaded_bytes` equals `transform_input_bytes` here: these
are logical host transform inputs, **not measured PCIe or network bytes**.
`delta_s` is call wall time; the verification harness separately measures the
subsequent synchronous prompt through its first token.

This uses a version-sensitive internal weight-free ABI and a global eager
compile context. Only one owning engine per device per process is allowed,
including after `close()`; replace its worker process for another owning model.
KV is host-resident. Initial compilation has substantial host-memory and time
costs. There is no `atom-constants` HTTP/prepared-config factory yet. See the
[REA and device evidence](../experiments/2026-10-23-resident-constant-update.md)
for the exact tested scope; this is not arbitrary byte patching of compiled files.

The full Llama-3.1-8B B1/S64 test changed one 117 MB group four times, with all
generated tokens equal to fresh checkpoint loads and the same native runtime
and program retained. Decode steps measured 68.7 ms; live onggi analysis through
the first updated token took 2.66–2.74 s. Initial compilation/loading took
151–168 s. B1/B4 constant sharing was additionally checked on a bounded matrix
stack; the full 8B update test used B1 only.

```bash
RBLN_WEIGHT_FREE=1 PYTHONPATH=src python experiments/verify_npu_delta.py \
  --backend atom-constants --base "$BASE" --new "$NEW" --store "$STORE" \
  --onggi "$ONGGI" --cache "$PRIVATE_CACHE" --batch 1 --sequence 64 \
  --tokens 8 --rounds 2 --out atom-constants.json
```

### Existing device-input and bundle paths

```bash
PYTHONPATH=src python -m pytest -q tests/swap

RNGD_MAX_SHAPES=8 python -m rngd_pd.swap.bench_rngd all \
  --ckpt "$CHECKPOINT" --group 8 --chip 0 --cache "$CACHE" \
  --variants B4,S256 B8,S256 B4,S512 --P 8 --n 8 --out rngd.json

python -m rngd_pd.swap.bench_atom_bundles --bundle "$ATOM_BUNDLE" \
  --variants B1,S512 B8,S512 --tokens 8 --rounds 3 --out atom.json
```

The harnesses compare token IDs with fresh loads and exit nonzero on a mismatch.
They distinguish program selection, preparation and request-to-first-token
latency. ATOM uses a generation streamer to timestamp the first generated token;
RNGD timestamps the first prompt result before continuing generation. Raw
artifact paths and arguments belong in local run records, not public evidence.
See the experiment record for measured hardware, versions and limitations.

For an end-to-end onggi update and fresh-token comparison on either chip:

```bash
python experiments/verify_npu_delta.py --backend rngd \
  --base "$BASE" --new "$NEW" --store "$STORE" --onggi "$ONGGI" \
  --cache "$CACHE" --batch 4 --sequence 256 --out rngd-delta.json
# ATOM: use --backend atom --batch 1 --sequence 64 and its own cache directory.
```

The harness checks both directions over two rounds, fresh-destination token
parity, unchanged ATOM allocation identities, and absence of recompilation.
It also requires the changed fixture to produce different tokens from the
base, preventing a non-discriminating prompt from proving an update.
