# Controlled resident delivery: results and limits

Issue #47 / PR #48. Native runtime baseline db2dd259, native harness829573e, Onggi
core05c9760; SDK2026.4.0, Torch2.10.0+cpu, Go1.25.1. All five frozen main
cohorts r02-r06 completed, alternating order, with no failed/retried/excluded
native main sample. Smoke and functional controls are separate.

All arms keep one base runtime/weight upload resident, initially with only the
two base runnable files. The target is B4/S512/group2 from B4/S256/group2.
Each stream has eight prompt and eight generated token IDs. Same-target native
parity, serving buffer/base-program identity, one initial upload and zero
compilation pass all 20 main transitions (10,200 original native calls).

## Primary n=5 measurements

Seconds; median [inclusive IQR] for total, component medians separately.

| Target / delivery | Demand → first token | Delivery | Full-file SHA | Stored pack payload |
|---|---:|---:|---:|---:|
| Graph / colocated POSIX copy | 0.402495 [0.008617] | 0.006648 | 0.001907 | N/A |
| Graph / CSI default-policy fallback | 0.416117 [0.002011] | 0.023101 | 0.001973 | 235,221 B |
| Full / CSI reconstruction | 67.576389 [0.307710] | 57.890721 | 9.251679 | 10,702,211,666 B |
| Full / CSI rebase | 9.881452 [0.289921] | 0.314540 | 9.165201 | 197,911 B |

The full bundle has 16,061,585,097 logical bytes; the graph-only bundle has
1,003,916 B. The default graph donor is refused because 224,097 B (22.3%) keep
their offsets, below the unchanged 50% minimum. This intended fallback is valid;
it is not a rebase success. Full rebase proves actual takeover, positive retained
bytes and inode continuity. CSI reconstruction keeps its unused independent
donor intact. Graph copy offers no donor and makes no claim.

Graph-only residency already avoids checkpoint movement. CSI does not improve
the local graph-copy baseline here. Full-bundle rebase reduces delivery work,
while whole-file verification dominates its residual validated total. These
controls isolate delivery with native residency held fixed; historical E6's
restart comparison cannot be relabeled as the current factorial control.

First token is reconstructed from generation entry plus returned native prompt
TTFT. All full-file hashes are inside the demand interval. The remaining seven
generated tokens are checked but excluded from first-token latency. Component
medians do not add to the median total. This is colocated POSIX/real CSI/native
decode, not HTTP readiness, a production FXB baseline, or a network benchmark.
Empty application chunk cache/no peers does not mean cold shared OS page cache.
Pack payload excludes metadata and is neither total filesystem reads nor NIC
traffic. Logical target size is not observed writes or DMA.

## Setup and memory are separate costs

| Arm | Harness entry → first token, median s | External CPU donor preparation, median s |
|---|---:|---:|
| Graph copy | 16.690 | N/A |
| Graph CSI | 17.465 | 0.057816 |
| Full reconstruction | 111.776 | 72.390 |
| Full rebase | 59.982 | 67.282 |

CPU CSI preparation includes real publish, full hash/inode verification and
unpublish. Both whole-bundle controls receive equivalent independent donors;
the no-rebase control does not need that donor in a deployment. These costs are
not free and are not a single measured publication-to-ready clock. Scheduling
and gaps between jobs are not included. Do not sum marginal medians or imply
one-off preparation-inclusive superiority. A pre-existing idle artifact is the
operating prerequisite for useful base consumption.

Base-ready host PSS medians are 19.53–19.55 decimal GB. Process lifetime HWM,
RSS/PSS and artifact inode/block witnesses are retained. They are not device
peak memory. Device allocation peak is unavailable. Container memory.peak spans
sequential trials and must not be attributed per arm. Donor logical/allocated
bytes are in raw inventories; roots may share inodes and must not be summed.

## Ownership and context functionality

The separate s02 full-csi-active case constructs the native owner from the
actual published base checkpoint and graph mount. With its lease held, CSI
refuses that busy base and publishes a separate target; files/inodes/hashes and
a native base request remain valid. The same owner then switches to target and
closes before unpublish. This n=1 observation uses 765 native calls. The earlier
s01 active-graph control alone did not establish checkpoint ownership.

Capacity c01 runs fresh S512 and resident S256→S512 in separate processes in
one single-card Job. Four 257-token prompts plus eight generated IDs match the
fresh S512 target. S256 rejects that request before native calls. Resident
weights/base programs remain intact; fresh/resident calls are 4488/4743, with
zero compilation and one shared upload per process. This n=1 correctness case
does not supply a capacity latency distribution or model-quality evaluation.

## Provenance and failures

All eight native Jobs (two smoke/ownership, five main, one capacity) succeeded
and were UID-cleaned with Pods absent. Every Job requests one RNGD and has TTL
and active deadline. Exact operational records include source/binary pins,
submitted Jobs, allocation identities, configs, actual CSI logs, RPCs, leases,
mounts, memory snapshots and cleanup receipts. The package contains 262 exact
cohort worker files plus 141 operational files, with compressed exports/hashes.
Model payloads/vendor binaries are absent; offline checks verify worker hash
witnesses and cannot rehash absent models.

CPU build01's early source-transfer error and r01's missing container
`/usr/bin/time` are retained. Both failed before native allocation. Existing
host tools were then mounted read-only and actual synthetic fixture02 passed
before new r02-r06. No native outlier is discarded. Synthetic fixture writes
are instrumentation checks, not model measurements.

Run the three verification commands in the data README. Mutation tests reject
byte/archive corruption, false native counts, false busy-base claims and invalid
time partitions even with recomputed export hashes. CI reproduces the complete
n5 summary and verifies context/source/UID evidence. Loaded-service drain/queue
experiments and CPU same-cache/chunking studies remain separate work.
