# CCGrid 2027: per-level reconfiguration and resident-update experiments

Status: E6 three-arm smoke passed; the30-trial matrix is running and headline
repetitions are not complete. E8 uses
the corrected same-target fresh-load criterion and is queued after E6.
Tracking: [E6 #34](https://github.com/ai-computing/rngd-pd/issues/34),
[E7 #35](https://github.com/ai-computing/rngd-pd/issues/35),
[E8 #36](https://github.com/ai-computing/rngd-pd/issues/36),
[optional E9 #37](https://github.com/ai-computing/rngd-pd/issues/37).
Implementation/method ledger: [plan](../docs/superpowers/plans/2026-10-30-ccgrid-npu-experiments.md).

Partial evidence: [preparation and source/input hashes](data/ccgrid-npu-20261030/preparation.json),
[fresh references and preserved cross-shape observation](data/ccgrid-npu-20261030/references-and-cross-shape-observation.json),
[published sample index](data/ccgrid-npu-20261030/publication-index.json).
Each smoke arm has n=1 and is excluded from headline aggregates. Files are
path-sanitized copies with original private-result hashes; no native/model
payload is downloaded or published. The final n5 collection will be separate.

The [raw-linked E6 collection](data/ccgrid-npu-20261030/e6-collection.json) and
[summary](data/ccgrid-npu-20261030/e6-summary.json) are explicitly incomplete
until all30 trials finish. At this publication milestone20 main samples are
verified; no group has five repetitions. The
[frozen matrix plan](data/ccgrid-npu-20261030/e6-frozen-matrix-plan.json) fixes
the blocked order before measurement. Published samples include their actual
physical-card assignment and verified cleanup evidence. Numeric values in the
normalized collection were checked against the unredacted original records.
The submitted Job requests4 CPUs/64GiB and limits12 CPUs/128GiB, with RNGD1.
CPU cores and background node load are not exclusively controlled. The fixed
SDK venv is mounted read-only; ephemeral Ubuntu dependency bootstrap counts in
Pod time. These native Pod timings do not describe a prebuilt production image.

## Questions and fixed comparisons

E6 measures an actual B4/S256→B4/S512 switch on the same Llama-3.1-8B
checkpoint and RNGD SDK2026.4/Torch2.10/group2 stack. Complete artifacts include
the checkpoint, precompiled graphs and configuration. The three arms are full
Onggi pull+engine subprocess restart, base-aware delta apply+engine restart, and
the same delta apply+resident weight reuse. This is a native generation benchmark,
not HTTP/live gateway service. No vendor artifact internals are parsed.

Every arm first starts the same base engine and reaches a drained barrier.
The immutable checkpoint used by that live engine is separate from an independent
consumable node base copy; Onggi may move the latter. Base copy/load and cache
setup stay inside Pod-creation time. Report both actual Kubernetes
`creationTimestamp`→first native-generated target token and same-host monotonic
reconfiguration-trigger→first target token. Keep scheduling/setup, file pull,
shutdown/startup, host preparation/upload, graph/table preparation and prompt
processing separately. Resident initial loading is not silently free.

Declare five independent trials per arm/cache state in interleaved blocks.
`empty` and `prepopulated` refer only to the node chunk cache. Target native
graphs are first-use in E6; OS page cache remains uncontrolled/shared. The POSIX
store is colocated with the node: store-read bytes are not network traffic.
Clock synchronization observations and any unknown wall-clock error bound stay
with Pod time; the monotonic trigger interval uses a shared host clock.

E7 separately changes sparse checkpoint rows and compares production inplace
updates, `group` replacement of entire touched stacked tensors, and restart.
Planning is recorded separately before base loading. Record update time,
plan time, first token and logical weight payload bytes; these are not physical
PCIe/DMA counters. Original raw `apply.uploaded_bytes` and row plans are retained.
Full checkpoint hashes come from the frozen external input manifest; the harness
does not rehash multi-GB shards inside the update interval.

E8 measures B4/S256→B4/S512, B4/S512→B8/S256 and B8/S256→B4/S256.
Five fresh-engine trials per direction in each graph state (`first-use` and
`prepared`) keep preparation history separate. Prepared state actually executes
and checks the target, returns to a checked base and then starts the switch
timer. Projection/LM buffers and already loaded graph identities must remain
unchanged. E9 is optional after the RNGD experiments; no vendor-versus-vendor
performance comparison is intended.

E8 correctness compares the switched/resident output with an independently fresh
load of the **same target variant**, requiring exact equality of every integer
token ID. It does not require B8 to reproduce B4's greedy continuation: different
batch tactics or BF16 accumulation order can change that continuation. Dense
logit/activation bitwise equality is not claimed by a token-ID comparison.
The earlier B8 cross-shape gate remains as its original failed attempt record;
the observed divergence position is recorded separately. This criterion change
was explicitly requested by the coordinating paper session before E8 repetitions.

## Reproduction interfaces

Run only on the owner's allocated card with the site's matching SDK/cache and
immutable input manifests. Never infer an allocation from an arbitrary chip ID.
The owner uses one `cdx-` Kubernetes Job at a time with RNGD request/limit1,
`ttlSecondsAfterFinished`, `activeDeadlineSeconds`, UID-guarded cleanup and
independent absence checks. Off-card CPU graph construction has separate records;
compilation is forbidden in the measured runtime paths.

```sh
export PYTHONPATH=src RNGD_MAX_SHAPES=8
python bench/ccgrid_e6.py trial --config trial.json --out new-e6.json
python bench/ccgrid_e7.py sample --ckpt BASE --target TARGET --ranges ranges.json \
  --variant B4-S256-g2 --cache CACHE --chip ALLOCATED_ORDINAL \
  --reference fresh-reference.json --host-workers 2 --intra-op 4 \
  --mode inplace --out new-e7.json
python bench/ccgrid_e8.py sample --ckpt BASE --cache CACHE --chip ALLOCATED_ORDINAL \
  --base-variant B4-S256-g2 --target-variant B4-S512-g2 --graph-state first-use \
  --reference shape-reference.json --host-workers 2 --intra-op 4 --out new-e8.json
python bench/ccgrid_summary.py collection.json --out new-summary.json
```

Script help/docstrings specify the exact configuration/reference schemas.
All result files use exclusive creation. Repeat attempts get new run directories;
failed outputs survive and are never overwritten. Reference continuations are
generated by independent fresh target workers, and every target/base output is
checked against them. No allocator retries or card-wait subtraction are hidden.

## Evidence and validity

The future collection links every original sample by SHA256 and retains its
source revision, environment, checkpoint/artifact identities, real prompts and
tokens, status, validity and phase samples. The analysis script rehashes linked
raw files and reports all attempts, valid n, median and inclusive linear IQR.
Any failed sample, missing metric or n<5 prevents a group from qualifying as a
headline. Additional corrective attempts remain separately labeled.

Native results and aggregate links will be added after the smoke and measurement
gates. CPU tests establish harness behavior only. The paper session owns
`data/NUMBERS.md`; this experiment does not modify that ledger.

The final harness CPU suite passed450 tests, skipped8 environment-dependent
cases and passed4 subtests. A single independent review found two Important
issues: lost E7 evidence on late native failure and mixing graph preparation
states in summaries. Both were reproduced and fixed before native repetitions.
E7 additionally preserves primary/cleanup errors, verifies original native output
devices and supports command-scoped Git provenance from root-owned Jobs.
E8 also verifies the allocated device ordinal through the same recursive output
guard. After that correction the local full CPU suite passed451 tests, skipped8
and passed4 subtests; exact-source CI bbceb17 passed457, skipped2 and passed4
subtests. These checks are not native E8 measurements.
