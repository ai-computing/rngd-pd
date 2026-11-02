# Controlled resident delivery experiment — issue #47

This experiment asks how artifact delivery changes first use of a new compiled
shape while one unchanged native `SwappableDecode` keeps its shared weights and
base programs resident. It changes only experiment files. The native runtime,
E6 observer/helpers and Onggi's eligibility threshold stay unchanged.

| Arm | Target artifact | Delivery | Base donor |
|---|---|---|---|
| `graph-copy` | `cache/graphs/` plus `variant.json` | Colocated POSIX `shutil.copytree`, then actual read-only bind | None; setup savings recorded |
| `graph-csi` | Same exact graph-only target | Actual CSI NodePublish | Independently materialized, unpublished, idle at least 1 real second |
| `full-csi-no-rebase` | Whole checkpoint/cache bundle | Actual CSI NodePublish, `--rebase-idle 0` | Independent idle base present; ordinary Onggi reconstruction control |
| `full-csi-rebase` | Same whole target bundle | Actual CSI NodePublish, `--rebase-idle 1s` | Independent idle base eligible for claim |
| `graph-csi-active` | Same graph-only target | Actual CSI NodePublish, `--rebase-idle 1s` | Only donor remains CSI-published and the native base engine reads its mount |

Graph copy measures local colocated copy, not a cloud object fetch. The whole
no-rebase arm is an Onggi reconstruction control, not a traditional copy arm.
Graph-only artifacts contain no checkpoint, symlinks or unrelated files. The
four timing arms read the same external immutable checkpoint. Graph-only rebase
may refuse a donor because stripping weights can lower same-offset reuse below
50%; an observed refusal and full reconstruction is valid. Never lower that
threshold. Whole rebase must prove a claim in smoke or the main matrix stops.

## Invocation and allocation

Run the complete fixed matrix inside one bounded Kubernetes Job with one
exclusive allocated card. Launch every trial as a fresh CLI process and wait for
its exit before starting the next trial; this releases process-owned allocations.
The parent records actual Pod metadata, resource allocation, environment,
revision/dirty state and binary/source hashes. A successful CPU config check
does not establish device, CSI privilege or measurement validity.
Before engine construction the runner independently requires Linux, exactly
one kubelet-allocated RNGD, an unambiguous allocation UUID, a matching
`/run/swap/rngd` init marker and the actual resolved Torch ordinal matching
`chip`. It rejects an imported runtime outside this checkout and records native
Python source fingerprints and installed Torch/SDK package versions.

```sh
export PYTHONPATH="$PWD/src"
python bench/ccgrid_resident_delivery.py --check-config --config /owned/trial.json
python bench/ccgrid_resident_delivery.py trial --config /owned/trial.json --out /owned/new-result.json
```

`--out` must not exist. `work_dir` must be an absent absolute path independent
of all existing inputs. Only trial-owned mounts and state may be configured.
Linux mount privileges in the current container/mount namespace and working SDK
and driver mounts are prerequisites. There are no retries or allocator waits.
The outer Job's active deadline bounds all trials; CSI calls each have a 300s
deadline and server start/stop have local bounds.

The required JSON fields are:

```json
{
  "trial_id": "smoke-graph-copy-01",
  "arm": "graph-copy",
  "phase": "smoke",
  "immutable_checkpoint": "/inputs/checkpoint",
  "base_cache_dir": "/inputs/base/cache/graphs",
  "base_bundle_source": "/inputs/base-graphs",
  "target_bundle_source": "/inputs/target-graphs",
  "base_ref": "sha256:<64 lowercase hex digits>",
  "target_ref": "sha256:<different 64 lowercase hex digits>",
  "store": "/owned/store",
  "onggi_binary": "/owned/bin/onggi",
  "csi_client": "/owned/bin/csi-call",
  "reference_file": "/inputs/reference.json",
  "base_variant": "b4-s256-g2",
  "target_variant": "b4-s512-g2",
  "chip": 0,
  "host_workers": 2,
  "intra_op": 4,
  "work_dir": "/owned/new-trial-work",
  "provenance": {}
}
```

The placeholder refs above must be replaced by actual immutable refs. Variant
parsing uses the real `Variant.parse`, rather than assuming case/format.
Reference JSON supplies `base_prompts`, `base_tokens`, `prompts`, and `tokens`:
each is exactly B equal-length nonempty lists of nonnegative integer token IDs,
within the relevant S context. Tokens must come from independently generated
fresh base and same-target reference runs; the parent preserves their provenance.
The runner preserves the entire supplied config verbatim and its SHA-256.

## Donor preparation and CSI

For local preparation, the runner starts a child `onggi csi` with a fresh prep
chunk cache, actual NodePublish of base, full inventory/hash/inode verification,
and NodeUnpublish for idle arms. Active-base E keeps the mount and lease. It
stops that server after RPC completion and starts a new server with the same
materializer data directory and an empty demand chunk cache. It waits 1.05 real
seconds; it never fabricates timestamps. Server argv are retained exactly:

```sh
onggi csi --store STORE --node-id cdx-exp --endpoint unix:///owned/work/csi.sock \
  --data-dir /owned/work/materializer --cache /owned/work/demand-cache \
  --workers 8 --pulls 1 --gc-every 0 --evict-after 0 --rebase-idle 1s
```

The no-rebase control uses `0`. The client performs actual NodePublishVolume or
NodeUnpublishVolume and emits JSON, with nonzero exit on RPC failure:

```sh
csi-call --socket /owned/work/csi.sock --operation publish \
  --volume-id target --target /owned/work/target-mount \
  --ref sha256:ACTUAL_DIGEST --timeout 300s
```

To avoid staging a 16GB full donor while occupying a card, idle CSI arms may use
optional `prepared_materializer` and `preparation_receipt` absolute paths.
Each trial needs its own unique task-owned state directory, already prepared by
real CPU CSI publish/unpublish, with the preparation server stopped and no
leases, volumes or in-flight RPCs. E and graph-copy reject external prep. No state
is copied into the trial: the demand server uses the original data directory so
inode identity remains meaningful. The receipt is independently retained as:

```json
{
  "status": "completed",
  "materializer": "/owned/unique-prepared-state",
  "base_ref": "sha256:ACTUAL_BASE_DIGEST",
  "inventory": {
    "variant.json": {"size": 100, "inode": 123, "device": 42, "blocks": 8, "sha256": "ACTUAL_FILE_SHA256"}
  },
  "preparation_started": {},
  "preparation_finished": {},
  "commands": [],
  "logs": [],
  "preparation_costs": {}
}
```

`inventory` must contain every base bundle file with actual metadata and SHA-256;
the example contains one abbreviated illustrative entry. Additional receipt
fields survive verbatim. The runner checks receipt identity/status, idle state,
actual donor inventory and complete hash/inode witnesses before server start.
Preparation costs remain explicit external costs excluded from harness-entry
timing. Local donor preparation (including full base materialization) remains
inside the harness timer and outside the demand endpoint; neither is free.

## Measurement boundaries and validity

The native observer is installed before engine construction. The runner loads
and generates the base reference, synchronizes the original engine, verifies
base parity/calls and absence of the target variant, then stamps a zero-inflight
barrier and demand. Delivery starts at demand. Delivery-only ends when actual
NodePublish or copy plus read-only bind returns. The runner confirms the exact
mount is read-only in its own `/proc/self/mountinfo`, then hashes every delivered
file against the exact expected inventory. These full 16GB hashes intentionally
remain inside demand-to-first-token for whole-bundle arms.

After hash completion the runner changes only `engine.cache_dir`, then uses E6's
unchanged `generate_target` helper for native swap/prepare and generation. It
requires target token parity, expected original native calls, exactly one shared
upload, zero compilation, unchanged shared tensor object/data pointers, and
unchanged base graph object IDs. Original counters and program/allocation
identities, full generation completion, and native first-token timestamps remain
in the record. The primary endpoint uses the engine's native prompt TTFT added
to generation start; it excludes the rest of the continuation.

Metrics partition delivery, full hash validation, hash-to-generation and
prompt-to-first-token, and give their demand-to-first-token total. Harness
entry-to-first-token and excluded pre-demand setup expose local preparation,
base engine construction/upload, input hashing and warmup. Externally prepared
donor costs are separately retained in the receipt. OS page cache is shared and
uncontrolled; empty chunk cache is not a claim of globally cold I/O.

For active-base E, lease/inode/hash preservation is checked after target
delivery, followed by another real native base generation before switching.
Its extra base calls are recorded separately and included in total counters.
The preservation request and checks are included in E's demand endpoint.

Every CSI arm requires the exact target's successful `materialized` PullStats
JSON and a logical byte count matching the inventory. Whole rebase additionally
requires a successful takeover log for the configured base, positive bytes left
in place, donor directory removal and reused file inode continuity. Active-base
E requires an actual busy-donor refusal log and intact base files; no-rebase
requires its donor to remain. Inspect retained logs together with these
filesystem witnesses when accepting smoke. Never infer byte writes or DMA from file size/blocks.
Record refusal/fallback rather than relabeling it as rebase. The parent must stop
main if whole rebase smoke cannot establish a claim. All arms retain exact child
logs, RPC JSON/stdout/stderr/argv, lease/volume state, mountinfo, file inodes and
blocks, process RSS/PSS/HWM and cgroup memory snapshots at base-ready, delivery,
target completion, close and cleanup.

Every result starts failed/invalid. A completed/valid result requires native
gates and cleanup. Cleanup closes the engine before any target/base unpublish,
unmounts only its own bind, and stops the CSI server. Failed attempts retain
partial records, all child logs and large work/state trees for export; the
runner never automatically deletes them. A failed trial's primary metrics are
cleared, while the stage timestamps and partial evidence remain. Report process
status, native correctness, delivery/claim mechanism and measurement validity
separately. Do not pool smoke with measurement repetitions or drop failures.
Cleanup also verifies that neither owned mount nor any lease/volume file remains.

## CPU verification

```sh
PYTHONPATH=src python -m pytest -q tests/swap/test_ccgrid_resident_delivery.py
PYTHONPATH=src python -m pytest -q
```

CPU tests cover immutable refs, fixed conditions, path independence, exact
inventory/hash errors, reference shape, timing endpoints and the actual mount
evidence parser. They do not establish native or real CSI smoke success.

## Active checkpoint ownership control

`full-csi-active` extends the bounded ownership observation to the checkpoint
itself, not only the graph files. Prepare an independent full base through real
CPU-only CSI publish/unpublish and retain its exact receipt. In the native Job,
publish that existing base again through NodePublishVolume, keep its live lease,
and construct the owner from `base-mount/checkpoint` and its graph cache. Target
publication must exclude this busy base and reconstruct separately. Verify the
base files, lease and a native base request after target materialization, then
perform the same resident target switch and close the owner before unpublishing.
This is a functional n=1 control, separate from the four-arm n=5 timing matrix.
The earlier graph-only active control does not establish checkpoint ownership.

## Frozen main and excluded attempts

The s01 functional smoke used an immutable graph cache containing all original
compiled runnable files; its target variant was absent from the native owner.
It is not pooled with the main timing samples. Main r02-r06 uses only the full
base artifact's two-file graph cache for initial load: the target decode
runnable is absent there. Both graph-only delivery arms provide exactly the
same target graph bundle, and all four arms then use the same resident native
switch. Odd cohorts use the table order and even cohorts its reverse. No
default eligibility threshold is changed. s02 separately verifies actual
full-checkpoint mount ownership and is a functional n=1 observation.

r01 stopped during CPU setup, before any native allocation: `/usr/bin/time`
was absent inside its slim container. Its failure is retained. The excluded
synthetic fixture02 uses existing host time/strace binaries mounted read-only
at their expected paths and passes actual CLI/syscall checks. Main starts at
r02 with unchanged native harness829573e and runtime baseline db2dd259. CPU
preparation, card allocation, native execution, export verification and cleanup
are separate stages; the next cohort starts only after the prior card and its
large owned payloads are released. The synthetic fixture is not model data.

Verify byte-exact archives, source fingerprints, native gates, ownership and
time partitions before reporting any n=5 aggregate:

```sh
python experiments/verify_resident_delivery.py --cohorts s01 s02
python experiments/verify_resident_delivery.py --cohorts r02 r03 r04 r05 r06 --summary
```

The second command requires all five main cohorts; it cannot summarize smoke
as repetitions or silently discard a failed/missing sample. First token is
reconstructed from generation-call entry plus returned native prompt TTFT.
Component medians need not add to the median total. Export checks verify
recorded model hashes and worker witnesses; model payloads are absent from Git
and cannot be rehashed by this offline verifier.
