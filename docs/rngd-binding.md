# RNGD device binding: supported subset and evidence boundary

`RngdBinding` implements the ADR-0010 B2 transaction interface for a bounded,
lossless subset of the existing `SwappableDecode` serving layout. The default
factory owns the real serving engine; an injected CPU factory is a transaction
test double. One native run passed the unchanged neutral B2 kit for identical,
interior projection-row and LM-row fixtures, with actual serving readback,
retained undo, independent fresh-token parity and a silent-corruption gate.
This is bounded-subset n=1 conformance evidence, not an n≥5 timing result.
The original coarse E7 sparse range correctly requires reload because it also
touches unsupported host/scaled bytes; that failed attempt is preserved.
See the [native report and exact raw evidence](../experiments/REPORT-RNGD-BINDING-20261030.md).

The adapter passes checkpoint byte ranges to existing serving tensors. It does
not map arbitrary compiled `.edf`, `.fxb`, or other opaque model bytes into
device addresses. onggi continues to supply vendor-neutral switch plans; the
safetensors/layout interpretation lives in this runtime.

## Exact units subset

Admission accepts a caller-verified artifact directory and its manifest digest.
The owner checks lowercase SHA-256 identity syntax, public `config.json`, the
complete file inventory and safetensors headers, duplicate tensor names, and an
optional matching `model.safetensors.index.json`. It does not recompute the
onggi manifest or prove that an arbitrary manifest names those files. That
verification remains the admitting caller's responsibility. See
[`rngd_binding.py:27–70`](../src/rngd_pd/swap/rngd_binding.py#L27-L70).

The normal factory requires the existing Llama-3.1-8B graph dimensions:
hidden size 4096, intermediate size 14336, 32 attention heads, 8 KV heads and
head dimension 128, with supported activation/bias settings. The layer count
must divide into the chosen group. The default binding variant is
`B4-S256-g2`; the selected variant must already have compatible graph-cache
entries. Load uses `allow_compile=False` and two host layout workers. See
[`rngd.py:142–177`](../src/rngd_pd/swap/rngd.py#L142-L177),
[`rngd_binding.py:154–181`](../src/rngd_pd/swap/rngd_binding.py#L154-L181), and
[`rngd.py:267–289`](../src/rngd_pd/swap/rngd.py#L267-L289).

Only these two-dimensional BF16 checkpoint tensors can become update units:

| Checkpoint tensor | Actual serving allocation and lossless mapping |
|---|---|
| `model.layers.i.self_attn.k_proj.weight` | `shared[group]["wk"][layer]`, checkpoint rows become columns of the resident transpose |
| `model.layers.i.self_attn.v_proj.weight` | Same mapping through `wv` |
| `model.layers.i.self_attn.o_proj.weight` | Same mapping through `wo` |
| `model.layers.i.mlp.gate_proj.weight` | Same mapping through `wg` |
| `model.layers.i.mlp.up_proj.weight` | Same mapping through `wu` |
| `model.layers.i.mlp.down_proj.weight` | Same mapping through `wd` |
| Untied `lm_head.weight` | Existing `wlm_d` chunks; checkpoint row order is preserved and vocabulary padding is untouched |

The layout cache admits a projection only when the actual tensor shape is
`(group, checkpoint_columns, checkpoint_rows)`. LM admission requires an untied
head used by the engine, matching column width and enough padded rows. Every
serving tensor checked by the owner must be BF16 on the declared device.
[`rngd_binding.py:22–24`](../src/rngd_pd/swap/rngd_binding.py#L22-L24) and
[`rngd_binding.py:121–152`](../src/rngd_pd/swap/rngd_binding.py#L121-L152)
define these checks.

Plans require the resident base manifest, the same file inventory, same-path
pairing and identical file sizes. Each non-`same` range must lie entirely in
admitted tensor data. Ranges round outward to complete checkpoint rows;
overlapping/adjacent row spans merge, and LM spans split at existing chunk
boundaries. More than 64 resulting units selects reload. There is no additional
changed-byte percentage threshold. Identical artifacts can produce a valid
zero-unit transaction. A range reported `elsewhere` still reads its replacement
bytes from the verified target, with `source: "target"`.
[`rngd_binding.py:212–275`](../src/rngd_pd/swap/rngd_binding.py#L212-L275)
contains the planner.

Changes to query projections, embeddings, layer/final norms, scaled or non-BF16
weights, tied LM heads, unsupported tensors, safetensors headers/gaps, config,
file sizes or pairing require reload. Query and norm paths are excluded because
the inherited runtime scales them; they are outside the byte-exact inverse map.
The underlying transforms are visible in
[`rngd.py:207–224`](../src/rngd_pd/swap/rngd.py#L207-L224).

`plan()` reads cached resident metadata and the supplied JSON, without target
file access, device readback or mutation. It returns a deterministic signed
update plan for the current load epoch. Missing/tampered proofs, a plan from
another load epoch and a no-longer-resident base are refused by `apply()`.
These are owner provenance checks, not authentication of external artifact
content. Before any write, apply additionally requires exact target config,
header, index and inventory metadata equality with the admitted base.
[`rngd_binding.py:207–210`](../src/rngd_pd/swap/rngd_binding.py#L207-L210) and
[`rngd_binding.py:339–357`](../src/rngd_pd/swap/rngd_binding.py#L339-L357)
show the proof and compatibility checks.

An unsupported plan returns `mode: "reload"` with a reason. Applying that plan
raises `NotImplementedError` before the update transaction; this adapter does
not silently implement full reload as B2 apply. The caller must perform an
explicit verified `load()` if it chooses the fallback. Invalid replacement
metadata preserves the existing owner; once replacement starts, the old engine
is closed before a new engine is created. A failed close quarantines the old
owner and blocks another load until an explicit `close()` retry succeeds.
[`rngd_binding.py:154–181`](../src/rngd_pd/swap/rngd_binding.py#L154-L181) and
[`rngd_binding.py:412–435`](../src/rngd_pd/swap/rngd_binding.py#L412-L435)
define replacement/release behavior.

## One persistent owner, whole-request admission

The JSON CLI is a proxy to an already-created owner, not a new model loader per
CLI invocation. The embedding process creates `RngdBinding` and
`binding_cli.create_server(owner, address)`, and owns both lifecycles. For
example, inside the allocated worker, with its verified graph cache mounted:

```python
from rngd_pd.swap.binding_cli import create_server
from rngd_pd.swap.rngd_binding import RngdBinding

owner = RngdBinding(cache_dir="/cache", variant="B4-S256-g2", chip=0,
                    expected_device="furiosa:0")  # explicitly allocated ordinal
try:
    with create_server(owner, "/run/rngd-binding.sock") as server:
        server.serve_forever()
finally:
    owner.close()
```

Use the backend name and ordinal matching the allocated device and SDK; the
owner rejects a mismatch. The server can instead run in a thread beside the
application's serving loop. CLI requests in separate processes all reach this
same owner:

```bash
PYTHONPATH=src python -m rngd_pd.swap.binding_cli --socket /run/rngd-binding.sock resident
# Binding command after the neutral kit's `--`:
python -m rngd_pd.swap.binding_cli --socket /run/rngd-binding.sock
```

The kit appends the operation and arguments. `conformance-load DIR MANIFEST`
calls the owner's normal load hook. The proxy supports all five ADR operations
plus this hook, uses bounded newline-delimited JSON, returns nonzero on an
owner error, and imports no native engine. A serial socket server dispatches
each request to the same owner; its Unix socket is created with mode 0600.
TCP is also available, without a separate authentication protocol. Closing
the server does not close the supplied owner. See
[`binding_cli.py:25–66`](../src/rngd_pd/swap/binding_cli.py#L25-L66),
[`binding_cli.py:87–132`](../src/rngd_pd/swap/binding_cli.py#L87-L132), and
[`binding_cli.py:163–199`](../src/rngd_pd/swap/binding_cli.py#L163-L199).

Serving code must use the owner's context for the entire generation or KV
session:

```python
with owner.request() as engine:
    tokens, ttft = engine.generate(prompts, n_new=8)
```

Holding it only around individual steps would permit an update between steps.
The context takes the owner lock and engine lock; apply/rollback use the same
locks and synchronize before mutation. An update increments the engine's KV
generation before a possible write, including a subsequently restored failed
transaction; old KV must be discarded. Direct mutation and the engine's legacy
`load_delta()`/`apply_delta()` are outside this ownership contract. The engine's
initial checkpoint handle stays open, while the binding records its newer
artifact identity separately. See
[`rngd_binding.py:193–205`](../src/rngd_pd/swap/rngd_binding.py#L193-L205),
[`rngd_binding.py:351–388`](../src/rngd_pd/swap/rngd_binding.py#L351-L388), and
[`rngd.py:389–400`](../src/rngd_pd/swap/rngd.py#L389-L400).

## Actual serving readback, journal and rollback

Apply stages every complete target row on the CPU before writing. It then
reads the old values from the actual `shared` or `wlm_d` serving allocation:
`tensor.cpu()` downloads the parent allocation first; row selection and inverse
transpose happen on the CPU. The undo source is neither the old artifact path
nor an unused device mirror. This works when onggi has patched the original
path in place or the previous checkpoint bytes are no longer available.
[`rngd_binding.py:277–303`](../src/rngd_pd/swap/rngd_binding.py#L277-L303).

Each attempted unit is journaled before its write is attempted. Replacement
rows are staged to the declared device and copied into a view of the original
serving allocation. After synchronization, the same serving allocation is read
back and converted to checkpoint row order; its SHA-256 must match the staged
target bytes. Only then does the owner publish the target manifest and retain
the successful undo journal. A later `rollback()` restores, synchronizes and
verifies every journaled unit before publishing the base identity; a second
rollback has no journal and fails. The journal is in-process CPU memory, not a
crash-recovery log. See
[`rngd_binding.py:305–330`](../src/rngd_pd/swap/rngd_binding.py#L305-L330) and
[`rngd_binding.py:358–410`](../src/rngd_pd/swap/rngd_binding.py#L358-L410).

A write/readback/publication failure or interruption restores and verifies all
attempted units, including a copy that partly wrote before raising. A failed
second apply retains the preceding successful journal if restoration succeeds.
If restoration or explicit rollback fails, the owner unpublishes the manifest
and unloads; a failed native release stays quarantined. This prevents admission
through the owner with a mixed serving model, without promising survival of a
fatal SDK/device failure.

`bytes_written` counts logical bytes copied into serving units. It excludes
temporary H2D staging, additional device copies and D2H; it is not a measured
DMA/PCIe counter. `readback_payload_bytes` accumulates the logical sizes of
whole serving parent tensors downloaded for undo and verification, and can
greatly exceed changed-row bytes. `readback_s` measures the `.cpu()` calls;
`initial_load_s` is separate from the update. These diagnostics do not measure
physical bus traffic. See
[`rngd_binding.py:288–299`](../src/rngd_pd/swap/rngd_binding.py#L288-L299) and
[`rngd_binding.py:332–337`](../src/rngd_pd/swap/rngd_binding.py#L332-L337).

## API and dependency audit

This is a source audit of the exact files hashed in the
[LOC report](rngd-binding-loc.json). Public Torch tensor operations in the new
adapter do not make the inherited serving engine a public vendor ABI.

| Boundary | Operations and exact source | Classification |
|---|---|---|
| Adapter, device readback | `Tensor.cpu()`, device/dtype checks, `numel()` and `element_size()`; [`rngd_binding.py:288–303`](../src/rngd_pd/swap/rngd_binding.py#L288-L303) | Public Torch tensor surface, exercised on actual serving parents by the production factory in the [bounded native B2 pass](../experiments/REPORT-RNGD-BINDING-20261030.md) |
| Adapter, device write | Tensor indexing, `.t().contiguous()`, `.to(device)`, and destination `.copy_()`; [`rngd_binding.py:305–320`](../src/rngd_pd/swap/rngd_binding.py#L305-L320) | Public Torch tensor surface; no direct Furiosa private import in the adapter |
| Adapter, host byte conversion | `torch.frombuffer(..., bfloat16)`, reshape/clone; contiguous uint8 view, NumPy bytes and SHA-256; [`rngd_binding.py:73–79`](../src/rngd_pd/swap/rngd_binding.py#L73-L79), [`277–286`](../src/rngd_pd/swap/rngd_binding.py#L277-L286) | Host Torch/NumPy plus Python stdlib; BF16 bits preserved without numeric scaling |
| Adapter to repository runtime | `shared`, `wlm_d`, `lm_name`, layout/group metadata, `_lock`, `_sync`, `_generation`, `_failed`, `swap()`/`close()`; [`rngd_binding.py:127–181`](../src/rngd_pd/swap/rngd_binding.py#L127-L181), [`351–435`](../src/rngd_pd/swap/rngd_binding.py#L351-L435) | Private/local engine contract; distinct from a private vendor SDK API |
| Inherited device setup/sync | SDK exports `set_fusion`, `set_mesh_kind`, `MeshKind`, `current_stream(...).synchronize()`; [`fsdk.py:31–33`](../src/rngd_pd/engines/fsdk.py#L31-L33), [`114–137`](../src/rngd_pd/engines/fsdk.py#L114-L137) | Top-level vendor API surface, still requires matching SDK/backend/device validation |
| Inherited graph construction | `CompileModule`, `TclModule`/`TacticKernelModule`, compiler config/tactic enums; [`fsdk.py:28–80`](../src/rngd_pd/engines/fsdk.py#L28-L80), [`decode_batch.py:39–71`](../src/rngd_pd/engines/decode_batch.py#L39-L71), [`114–159`](../src/rngd_pd/engines/decode_batch.py#L114-L159) | Existing SDK-dependent compilation/runtime setup; builder imports remain even when the binding forbids compilation |
| Inherited private SDK dependency | Compiler fallback imports `furiosa.torch._C.config.compiler`; [`fsdk.py:75–80`](../src/rngd_pd/engines/fsdk.py#L75-L80) | Explicit private namespace; not newly introduced by this adapter |
| Inherited Runnable cache | `module.runnable._inner.serialize()` and `ir.Runnable.deserialize`, then `Runnable`/`CompileModule` reconstruction; [`fsdk.py:55–64`](../src/rngd_pd/engines/fsdk.py#L55-L64) | `_inner` is explicitly private; native IR serialization/reconstruction is version-sensitive and is not claimed as a portable documented ABI |
| Inherited cache records | `module_weight`, `output_reconstructor`, zero-sized tensor and passthrough metadata; [`artifact_cache.py:56–80`](../src/rngd_pd/engines/artifact_cache.py#L56-L80), [`119–143`](../src/rngd_pd/engines/artifact_cache.py#L119-L143) | SDK record-layout coupling; trusted Torch pickle state and SDK/Torch identity checks, not a vendor-neutral format parser |
| Inherited legacy cache/device branch | `furiosa.torch.custom_ops`, `native_device`, `furiosa.native_torch.ir.Edf.deserialize`; [`fsdk.py:65–71`](../src/rngd_pd/engines/fsdk.py#L65-L71), [`artifact_cache.py:134–140`](../src/rngd_pd/engines/artifact_cache.py#L134-L140) | Older SDK-specific interfaces, separately retained; no assertion that this adapter revalidated them |
| Shared parser/policy | Public safetensors header reader and `onggi.switchplan.v1` validator in [`delta.py:56–179`](../src/rngd_pd/swap/delta.py#L56-L179); `Variant` in [`variants.py:12–35`](../src/rngd_pd/swap/variants.py#L12-L35) | Stdlib-only, vendor-neutral metadata/byte-space parsing; not device admission or artifact authentication |
| Host/model/runtime support | `safetensors.safe_open`, NumPy transforms, `ModelConfig.inv_freq`, graph specs, DSL strings and graph cache; [`rngd.py:93–130`](../src/rngd_pd/swap/rngd.py#L93-L130), [`201–289`](../src/rngd_pd/swap/rngd.py#L201-L289) | Existing host/runtime dependencies, counted separately from the adapter; `model_wired.py` and `vendors/` are not imported by this RNGD path |

## Reproducible LOC scope

Run `python experiments/analyze_rngd_binding_loc.py --out docs/rngd-binding-loc.json`.
The script imports no model/SDK modules. The checked-in JSON records every
explicit filename, its raw-byte SHA-256 and counts, plus each scope's source
identity hash. Physical and nonblank counts include comments and docstrings;
nonblank excludes only whitespace-only lines. Counts cover complete files,
including dormant helpers and CLI/benchmark functions within those files.

| Scope | Files | Physical | Nonblank |
|---|---:|---:|---:|
| Adapter only: `swap/rngd_binding.py`, shared `swap/binding_cli.py` | 2 | 634 | 567 |
| Shared parser/policy: `swap/delta.py`, `swap/variants.py` | 2 | 498 | 444 |
| Inherited RNGD runtime/support, explicit list below | 14 | 2,211 | 1,927 |
| Full RNGD implementation: union of the preceding three disjoint scopes | 18 | 3,343 | 2,938 |
| Whole `swap/` directory inventory, including ATOM and bench files | 14 | 3,710 | 3,292 |

The 14 inherited files are `rngd_pd/{__init__,_tv_patch,paths}.py`,
`rngd_pd/swap/{__init__,load,rngd}.py`,
`rngd_pd/engines/{__init__,artifact_cache,decode_batch,fsdk,model}.py`, and
`rngd_pd/kernels/{__init__,decode_batch,dsl_batch}.py`, all under `src/`.
`load.py` is conservatively included although the owner's contract excludes
its legacy update entry point. `_tv_patch.py` is also included for the optional
bare `_tv_patch` import attempted by `dsl_batch.py`; that import may not resolve
in a normal package-only path. Package initialization files are included.
`model.py` is counted in full even though this path uses its `ModelConfig`.

The whole-swap inventory is a separate directory budget, not the full RNGD
dependency closure and not an adapter size. It explicitly includes `__init__`,
`atom`, `atom_bundles`, `atom_constants`, `atom_inputs`, `bench_atom`,
`bench_atom_bundles`, `bench_rngd`, `binding_cli`, `delta`, `load`, `rngd`,
`rngd_binding`, and `variants` (`.py`). It is not added to the full-RNGD total.

SDK/package implementations (Torch, Furiosa, NumPy, safetensors, etc.), tests,
documentation, graph/weight artifacts and this analysis script are excluded.
ATOM-only modules and unrelated serving/connector/native-v4 modules are outside
the full RNGD scope. The report is an exact working-tree snapshot; rerun it and
update this table after source changes. Native conformance remains a separate
evidence gate from these counts and the CPU transaction suite. The
[evidence package manifest](../experiments/data/binding-20261030/package-manifest.json)
separately records native LF source hashes and exact/LF-equivalent comparisons
with this Windows working-tree/LOC snapshot; line-ending differences are not
silently treated as identical raw source bytes.
