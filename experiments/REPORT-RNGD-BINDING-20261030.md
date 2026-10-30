# RNGD B2 binding: native bounded-subset conformance, 2026-10-30

One native run passed the unchanged neutral B2 kit for identical, interior
projection-row and LM-row fixtures on one allocated RNGD: physical card 0,
`furiosa:0`, SDK 2026.4.0, Torch 2.10.0+cpu, variant `B4-S256-g2`. This is
**n=1 conformance evidence**, not an n≥5 timing result or support for arbitrary
checkpoint changes. The complete successful result is
[native-02/result.json](data/binding-20261030/runs/rngd-b2-native-02/result.json).

| Claim | Raw evidence |
|---|---|
| Three B2 cases pass every kit assertion, including pure planning, full changed-byte coverage, actual unit hashes, retained undo and second-rollback refusal | [kit report](data/binding-20261030/runs/rngd-b2-native-02/work/kit-report.json), [kit stdout](data/binding-20261030/runs/rngd-b2-native-02/work/kit.stdout.log) |
| Eleven generation gates match independent fresh tokens; 2,805 original compiled calls and 2,805 native output checks, zero compile attempts | `generations`, `native_counts`, `native_outputs_verified` in [native-02 result](data/binding-20261030/runs/rngd-b2-native-02/result.json) |
| Serving weight buffer pointers and program objects remain unchanged across apply/rollback; generation also preserves them | Each generation's `identities_before`, `identities_after`, `serving_allocations_programs_retained` and `generation_preserved_identities` in the same result; new explicit loads allocate a new owner |
| A one-byte silent BF16 write corruption is rejected by actual serving readback; attempted rows are restored and fresh-base token parity returns, with identities retained | `silent_corruption_gate` and final `silent_corruption_restored` generation in the same result |
| Public Torch strided copy, whole-parent `.cpu()` BF16 readback and exact retained-journal restoration work on the assigned card | [readback probe result](data/binding-20261030/runs/b2-readback-probe-01/result.json); before/restored hashes are identical |

The eleven generations are nine kit load/apply/rollback gates, one explicit
base-ID reload for the corruption case, and one restored-base gate after the
rejected write. The independent interior reference ran first in a separate
original-engine process and closed before the new owner was constructed; its
255 native calls are recorded separately from the binding observer's 2,805.
See [interior-fresh.json](data/binding-20261030/runs/rngd-b2-native-02/work/interior-fresh.json),
[its log](data/binding-20261030/runs/rngd-b2-native-02/work/interior-fresh.log),
and the actual combined [reference inputs](data/binding-20261030/runs/rngd-b2-native-02/work/refsactual.json).
Those rows retain the full prompts, generated tokens, original-source hashes,
checkpoint paths and assignment. The kit report retains complete base/target
file inventories and SHA-256 hashes; no checkpoint or graph payload is packaged.

| Successful planned unit, half-open checkpoint rows | Logical bytes per apply or rollback |
|---|---:|
| Identical | 0 |
| `model.layers.0.mlp.down_proj.weight@125:193` | 1,949,696 |
| `lm_head.weight@12037:12069` | 262,144 |

The preparation changed interior projection rows 128:192; actual onggi chunk
ranges round outward to the full planned rows 125:193. Equal row bytes within
those ranges remain covered. The final diagnostics record nine whole-parent
readbacks totaling **4,561,305,600 bytes** across verification/journaling and the
fault gate. This is readback payload accounting, separate from logical unit
writes. Neither quantity is measured DMA/PCIe traffic. H2D staging and repeated
device copies are excluded from the logical write count. Initialization and
raw `.cpu()` durations remain separate diagnostics; no speedup is inferred.

The exact [preparation record](data/binding-20261030/preparation/evidence/rngd-interior-preparation.json)
contains all six base/interior/source file hashes, source row payload hash,
and original onggi push/diff argv/stdout. Its [authentic interior switch plan](data/binding-20261030/preparation/inputs/interior.switchplan.json)
and [measured provider](data/binding-20261030/preparation/inputs/rngd-interior-fixtures.json)
match the native kit inputs. The original [provider](data/binding-20261030/preparation/inputs/rngd-fixtures.json)
and [coarse sparse plan](data/binding-20261030/preparation/inputs/sparse.switchplan.json)
are also preserved unchanged.

The failed first attempt is preserved in
[native-01/result.json](data/binding-20261030/runs/rngd-b2-native-01/result.json)
and its [kit report](data/binding-20261030/runs/rngd-b2-native-01/work/kit-report.json).
Its identical case passed all three generation gates. The original E7 sparse
chunk span, offset 1,050,641,786 and length 2,018,387, includes an equal embedding
tail and scaled layernorm bytes before `down_proj` begins at 1,050,690,864.
The planner correctly returned `reload`; applying that unsupported plan failed.
The successful attempt uses a newly prepared interior fixture, with no planner
weakening: both attempts record identical runtime `source_files` hashes.

The native runtime checkout was `a2ebc3281df4173a886bc6a035f2483eaa360bf5`;
the neutral kit was `71c37d26f3ec9bedc20c93b27c2cfde9dfd64d53`;
independent fresh generation used original runtime
`f5fa13377a3d967ba3d4022fb73ae59b1c87441c`. Actual device-run source hashes are
authoritative in the raw results. The Windows [LOC snapshot](../docs/rngd-binding-loc.json)
hashes raw working-tree bytes: inherited files can differ only by CRLF versus
native LF. [package-manifest.json](data/binding-20261030/package-manifest.json)
records exact and LF-equivalent comparisons without rewriting either source
or raw evidence. The adapter files match the native hashes exactly.

Reproduction harnesses are exact copies of
[run_rngd_b2.py](data/binding-20261030/harness/run_rngd_b2.py) and
[prepare_rngd_interior.py](data/binding-20261030/harness/prepare_rngd_interior.py).
The former's hash matches the successful native result. They require the
caller-verified fixtures, independent original source, precompiled cache and an
exclusive bounded allocation; running them is a new native experiment.
The new store may assign the same base bytes a different authentic manifest ID.
Exact measured copies of the [attempt-01 harness](data/binding-20261030/preparation/scripts/run_rngd_b2.py)
and [attempt-02 harness](data/binding-20261030/preparation/scripts/rngd-b2-native-02.py)
match their respective result hashes. The measured
[preparation script](data/binding-20261030/preparation/scripts/prepare_rngd_interior.py)
matches the preparation record. The exported
[public probe script](data/binding-20261030/preparation/scripts/probe_readback.py)
is preserved too; the probe result has no harness-hash field, so its source
provenance is the original preparation export entry rather than a claimed
result-field match.

The original gzip export archives, their per-file export manifests, Job
specifications, terminal API records and cleanup receipts are included for both
attempts; the probe includes its read-only export manifest and cleanup receipt.
The preparation directory is a [selected RNGD-only export](data/binding-20261030/preparation/selection-manifest.json)
with the original complete export manifest retained; the mixed ATOM preparation
archive and ATOM files are not duplicated.
All packaged bytes are marked `-text` to preserve their export hashes. Verify
them and regenerate the numeric summary without importing Torch or any SDK:

```bash
python experiments/verify_rngd_binding_evidence.py
```

The supported subset and inherited private SDK/cache dependencies remain
documented in [docs/rngd-binding.md](../docs/rngd-binding.md). The run proves
byte-exact restoration of planned serving units and fresh-token parity; it does
not add support for scaled/host tensors, arbitrary opaque compiled artifacts,
crash recovery, or a whole-model device digest.
