# RNGD resident row updates

Date: 2026-10-23. This work improves `SwappableDecode` checkpoint refreshes
while retaining its existing device-input decode graph and weight layout.

## Why the old path did extra work

The safetensors/Onggi planner already identifies changed source rows. The old
updater ignored them, transposed the entire changed projection, prepared another
complete transpose for rollback, uploaded a temporary device tensor, and copied
it into the resident allocation. In the current full-tensor baseline, new and
rollback transposes account for about 85% of the update interval.

The new updater stages only effective changed rows. Since projections are
transposed, checkpoint rows become resident columns. LM-head rows remain
contiguous and split at its packed chunk boundaries. Its existing allocation
and vocabulary padding stay intact. Embeddings and normalization retain their
existing conservative host/per-batch handling.

## SDK and REA investigation

The requested REAP feature-tracing/function-inspection method was reused.
REA 6.7.0 inspected three bounded native address windows, preserving artifact
hashes and Evidence IDs. Active SDK source confirms same-dtype host/device
`copy_` delegates to native `memory.copy` and waits with `run_blocking`.
HAL/sequencer and TP scatter APIs were investigated but were unnecessary for
the final change. [SDK facts and addressed observations](data/rngd-row-update-20261023/sdk.json)
separate source facts from the limitations of stripped native code. Selected
windows are excerpts, not complete function decompilation or raw-offset proof.

Bounded native probes used the actual `[8,14336,4096]` projection parent with
fusion 8/mesh 2. Each route checked the complete original parent, unchanged
neighbors, and a compiled consumer that used that parent. External/unlowered
operators were disabled; fallback was not accepted. Clean-lifetime observations:

| Operation | Logical patch | Transfer/copy samples |
| --- | ---: | --- |
| Whole layer, direct host copy | 117,440,512 B | 19.6–21.9 ms |
| Whole layer, temporary device plus copy | 117,440,512 B | 18.8–26.1 ms upload + 5.2–11.4 ms copy |
| Eight projection columns, staged device patch | 229,376 B | approximately 3.3–4.1 ms combined |
| Eight projection columns, direct host sequencing | 229,376 B | 22.9–26.6 ms copy |

Accordingly, contiguous full projections and LM rows copy directly from the
host, while sparse projection columns use small staged device patches. This
policy follows the measured layout rather than assuming direct copies always
win. Native consumers' checking intervals include readback/CPU-reference work;
they are not decoder kernel latency measurements.

Extra probes covered a nonzero-offset single-row source and an actual
`[32,4008,4096]` LM parent with a nonzero source offset and destination chunk 3.
The single-column consumer specifically selected the changed column and had
to change its output; all three repetitions and full-parent checks passed.
All 21 accepted samples, including the slower LM-offset sample, are in
[micro.json](data/rngd-row-update-20261023/micro.json). Probe compilation takes
38–66 seconds upfront and is outside resident-update timing; the production
engine reuses its existing EDFs.

## Safety and bounded work

Row bounds and real checkpoint shapes/dtypes are checked before native writes.
FP32 q scaling precedes BF16 conversion, preserving fresh-load rounding. Missing
row metadata retains whole-tensor behavior; header moves also select all rows.
After coalescing, more than four runs or at least half the rows selects one
whole effective range. This conservative policy bounds dispatch cost: review
reproduced a 256-copy fragmented plan before the fix.

New and rollback patches use the same effective ranges and are staged before
the first write. Failed/interrupted updates restore old weights, group/LM
references and per-batch state. Failed or interrupted rollback disables serving.
Old KV is invalidated before a potentially mutating transaction. Manually
supplied plans must completely describe changes; production callers should use
the planner against matching immutable checkpoint refs.

## Full-model protocol and scope

Use the existing furiosa-torch 2026.2.0, Torch 2.11.0+cpu and cached EDF pairing.
The graph is unchanged; this does not require a new decode compilation or a
different weight layout. The measured model is Llama-3.1-8B-Instruct revision
`0e9e39f249a16976918f6564b8830bc894c89659`, B4/S256, group 8, seed-0 eight-token
prompts and eight greedy generated tokens per stream.

Fresh references run in separate processes. Baseline and optimized resident
engines each perform `new → base → new → base` updates using actual live Onggi
diffs, immutable fixtures, the same allocated card, SDK, cache and CPU limits.
All timing samples, including each first update, must be retained. Logical
weight payloads are not physical PCIe/DMA counters. The host is shared and load
varies, so these are observed runs rather than an SLO or a controlled universal
speed claim. Whole-tensor measurements must not be used to imply a speedup
when system-state differences dominate them.

## Full-model results

Frozen baseline: `da85c80ba5f0eb979d32bbcbbbcd4b1de13b0c13`.
Frozen optimized code: `719c09b0f105782ffbf75ed3451ec9337d7545f2`.
Across three fixtures, both implementations completed all four updates:
**24 updates and 768 generated tokens exactly equal to fresh references**.
Each destination changed the generated output relative to the base. The new
updater retained original shared/LM allocations and compiled module objects;
preparation/graph creation was forbidden after initial setup. The old LM path
replaced its LM allocation, as expected.

| Fixture | Logical bytes, old → new | Update phase, old → new |
| --- | ---: | ---: |
| Complete down projection | 117,440,512 → 117,440,512 | 223–233 → 320–384 ms |
| Down projection, 64/4096 rows | 117,440,512 → 1,835,008 (64× smaller) | 271–327 → 7.1–10.3 ms |
| LM head, 8/128256 rows | 1,050,673,152 → 65,536 (16,032× smaller) | 476–618 → 1.20–1.77 ms |

The old `delta_s` starts after validation; the new one includes validation,
new staging, rollback staging and copy completion. Neither includes Onggi or
planning. Complete pipeline `total_s` has the same outer timing boundary in
both versions and includes Onggi, planning, checkpoint handling and the update:

| Fixture / version | Complete update, all four samples (s) | Median (s) | Through first token, all four samples (s) | Median (s) |
| --- | --- | ---: | --- | ---: |
| Full / old | 1.177, 0.669, 0.682, 0.692 | 0.687 | 1.400, 0.908, 0.904, 0.928 | 0.918 |
| Full / new | 2.100, 1.157, 1.099, 1.141 | 1.149 | 2.506, 1.526, 1.422, 1.423 | 1.475 |
| Sparse projection / old | 1.660, 0.777, 0.894, 0.855 | 0.874 | 2.031, 1.128, 1.235, 1.230 | 1.232 |
| Sparse projection / new | 1.373, 0.527, 0.482, 0.519 | 0.523 | 1.622, 0.903, 0.745, 0.761 | 0.832 |
| Sparse LM / old | 1.759, 1.028, 0.881, 1.008 | 1.018 | 2.213, 1.241, 1.078, 1.389 | 1.315 |
| Sparse LM / new | 0.902, 0.309, 0.289, 0.288 | 0.299 | 1.159, 0.638, 0.536, 0.509 | 0.587 |

Sample order is `new, base, new, base`, including the slower first update.
Through-first-token adds the measured first-token interval for an eight-token
prompt; it excludes subsequent generated tokens. Setup/publication are separate.
[llama8b.json](data/rngd-row-update-20261023/llama8b.json) preserves all unrounded
timings, phase breakdowns, fresh/resident tokens and 16 steady decode samples
per process.

Whole-tensor updates did **not** demonstrate a speedup. At process start, the
host's one-minute load rose from 31 for the full baseline to 101 for the full
candidate; initial resident load rose from 19.8 to 35.9 seconds, and planning
also slowed. Full-model steady B4 decode medians varied from 19.15 to 36.22 ms
across the six processes. These runs establish sparse update correctness and
lower logical payload/preparation cost, not faster decode or a universal latency
ratio. Onggi/checkpoint work dominates after the sparse update phase shrinks.

## Reproduction

Create matching immutable base/destination checkpoints and publish their refs
before timing. Changing a few values still updates complete selected rows,
not arbitrary individual bytes. The sparse projection fixture changes rows
`[0,64)` of layer-0 `down_proj`, seed 123 and scale 1.0. The LM fixture changes
rows `[12041,12049)`, seed 321 and scale 3.0. Setup, checkpoint copying and initial
store publication are outside resident-update timing.
The complete-projection fixture replaces all rows with seed 1 and scale 0.1.
The stock harness below expects the chosen fixture pair published under refs
`base` and `new` in its own store:

```bash
"$ONGGI" push --store "$STORE" --ref base "$BASE"
"$ONGGI" push --store "$STORE" --ref new "$NEW"
RNGD_MAX_SHAPES=8 GOMAXPROCS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python experiments/verify_npu_delta.py --backend rngd \
  --base "$BASE" --new "$NEW" --store "$STORE" --onggi "$ONGGI" \
  --cache "$EXISTING_CACHE" --batch 4 --sequence 256 --rounds 2 --out result.json
```

Run the frozen baseline and optimized revisions separately with matching
environments. Keep the matching PyTorch tree-spec cache; an SDK version alone
does not make EDF sidecar data portable. The API, phase metrics and payload
semantics are documented in [delta-swap.md](../docs/delta-swap.md).

## Validation

Independent review found a fragmented-copy regression and a missing native
source-offset validation case. The fragmented/dense fallback reproduced RED
then passed GREEN; both native source-offset probes passed. Tests also cover
FP32/BF16 projection transforms, LM chunk/padding retention, tied embeddings,
real safetensors planning, partial writes, interrupted rollback and stale KV.
After review fixes, local tests passed 296, skipped 6 and passed 4 subtests.
[Linux CI](https://github.com/ai-computing/rngd-pd/actions/runs/39918506613)
passed 300 tests, skipped 2 and passed 4 subtests on frozen `719c09b`.

Initial probes with an incomplete Pod toolchain or mismatched Torch/SDK paths
were rejected. A baseline instrumentation attempt made an invalid tensor-key
assertion; rendering that error invoked an unsupported tensor printer operation.
That attempt was rejected, all global Tensor monkeypatches were removed, and
the clean corrected baseline was retained. Vendor SDK/binaries were unchanged;
compiler prerequisites were installed only inside the owned Pod.

[provenance.json](data/rngd-row-update-20261023/provenance.json) records source,
SDK/native and Onggi hashes, fixture seeds/manifests/live diff ranges, resource
limits, host load, negative controls and cleanup. All 86 original files were
exported byte-for-byte and independently checked against remote SHA256/length
before private paths were sanitized for publication. Source snapshots match
the Git revisions above. Accepted logs contain no native fallback, CPU bridge,
panic or traceback; the REA excerpts do not replace these execution checks.

The owned Pod and namespace were deleted with UID preconditions. Final audits
found no owned processes or RNGD compute descriptors. Immutable fixtures,
stores and original evidence remain in the owned scratch workspace for audit;
the original provider/repo and vendor SDK were untouched. A live sample showed
npu0 utilization at 73.92%; the SDK memory counter stayed at zero even during
execution, so no measured HBM usage or physical transfer-byte claim is made.
Validation covers furiosa-torch 2026.2.0 and the existing matching EDF cache;
other SDK versions require their own validation.
