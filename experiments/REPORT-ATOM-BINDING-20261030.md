# ATOM public B1 binding: bounded native evidence, 2026-10-30

The default public `AtomSession` factory passed the unchanged neutral **B1**
kit for two artifact fixtures on one allocated ATOM, then completed five serial
reload/generation repeats. All seven binding loads produced the same eight
tokens as a separately loaded fresh `AtomSession`; the fresh owner was closed
before the binding loaded. The result records **120 completed native
`rebel.sync_runtime.SyncRuntime.run` calls and zero compilation attempts**.
The 15,176,307,063-byte precompiled graph retained the same source SHA-256.
This proves bounded public B1 behavior, not B2 device updates.

| Claim | Raw evidence |
|---|---|
| Two native B1 cases pass describe, resident identity, pure repeated planning, always-reload mode and wrong-base refusal | [native-04 kit report](data/atom-binding-20261030/runs/atom-b1-native-04/kit-report.json), [stdout](data/atom-binding-20261030/runs/atom-b1-native-04/kit.stdout), [empty stderr](data/atom-binding-20261030/runs/atom-b1-native-04/kit.stderr) |
| Seven binding generations match fresh tokens; 15 native calls each plus 15 fresh calls; no compiler entry called | `loads`, `fresh_reference`, `native_runtime_calls_completed`, `compile_attempts` in [native-04 result](data/atom-binding-20261030/runs/atom-b1-native-04/result.json) |
| Default public factory, no experimental constant attachment or private vendor handle use | Every load's `evidence.runtime_api`, `vendor_internal_apis_used`, `experimental_constant_attachment`, recorded runtime/compiled types in the same result; [exact measured harness](data/atom-binding-20261030/harness/run_atom_b1.py) |
| Compiled graph reused without recompilation | `graph.sha256`, `graph.final_sha256`, each load's `evidence.sha256`, compiler guards, and the [off-card compile record](data/atom-binding-20261030/preparation/evidence/atom-public-compile.json) |
| One requested/limited card, strict device exposure, unprivileged successful worker, exit zero and UID-bound cleanup | [Job](data/atom-binding-20261030/runs/atom-b1-native-04/job.json), [terminal API](data/atom-binding-20261030/runs/atom-b1-native-04/terminal-api.json), [cleanup receipt](data/atom-binding-20261030/runs/atom-b1-native-04/cleanup.json) |

The model was Llama-3.1-8B, 32 layers, B1/S64, with rebel-compiler 0.11.0,
Torch 2.11.0+cpu and host KV. The fixed prompt IDs were
`[1,42,256,1024,7,999,2048,13]`, followed by eight greedy tokens. The existing
public graph has embedded FP16 projection weights and its LM head on the NPU;
host `Tables` are loaded once and reused across the fresh and binding owners.
No fake runtime, device mirror, injected factory or experimental #28 constant
engine was used. Each public runtime creation returned a compiled allocation
estimate of 15,210,643,456 bytes; that SDK estimate is not measured physical
peak allocation or DMA traffic.

The two cases are `native-identical` and `native-metadata-revision`. Both use
the same compiled B1/S64 graph; the latter changes publisher profile revision
metadata. Their authentic onggi push/diff identities, complete four-file
artifact inventories, profile/loader/graph-sidecar bytes and switch plans are
preserved in [preparation](data/atom-binding-20261030/preparation/export-manifest.json).
The B1 kit loads the base for each case and checks the resulting reload plan;
it does not apply the target. These fixtures establish resident identity and
reload planning for identical and metadata-changing artifacts, not changed
weight correctness, device readback, rollback, or selective tensor updates.

## Five repeated reloads

These are five serial repetitions after fresh generation and the two kit
loads, on the same host/device and with reused host tables. The verifier
recomputes all statistics from the individual samples, using inclusive
quartiles. No sample is removed. The fresh load and kit loads are outside
these five-trial summaries.

| Interval, seconds | n | Median | IQR |
|---|---:|---:|---:|
| Owner load, including source validation/hash and old-runtime close | 5 | 20.048155 | 0.014296 |
| Admission metadata validation and source hashing, included above | 5 | 9.162043 | 0.002086 |
| Public compiled-model parse, from `session_timings.parse_s` | 5 | 10.491921 | 0.017418 |
| Public runtime creation, from `session_timings.create_runtime_s` | 5 | 0.353489 | 0.001783 |
| Prompt through first token, measured after load | 5 | 0.561485 | 0.000626 |
| Full eight-token generation, measured after load | 5 | 1.039721 | 0.001363 |
| Per-generation median decode step | 5 | 0.068308 | 0.000072 |
| Per-trial load plus post-load prompt TTFT, derived | 5 | 20.609640 | 0.019675 |

The 0.561485 s TTFT starts after `load` returns. The final row adds each
trial's load interval and measured prompt TTFT before computing statistics;
it is not the sum of independently summarized medians. It excludes harness
inspection between those intervals and is not a measured external trigger
or HTTP request through first response. Parse and runtime-create intervals
come from their own boundaries; subtracting summary medians is unnecessary.
Parsing and runtime creation are SDK-call wall times. Runtime initialization
may defer work into the first inference, so the 0.353489 s `create_runtime`
interval is not measured weight-upload or DMA latency. No device traffic
counters were captured; the record does not attribute the load time to NPU
transfer separately.
The repeated source reads and prior loads preclude a cold-cache claim. These
results establish no causal or cross-vendor speedup.

The fresh reference load took 10.869110 s, with 10.511827 s parsing and
0.357011 s runtime creation. Its post-load prompt TTFT was 0.558970 s.
That fresh load has no binding admission/hash interval and is not a matched
cold-load comparison. Initial public graph preparation was off-card: the
compile record reports 119.343441 s complete preparation, with zero runtime
creations. Graph preparation, checkpoint hashing and host-table loading are
outside the repeated reload rows.

## Device mapping and retained failures

The passing worker exposed exactly `/dev/rbln0`, allocated physical device 0,
and selected logical SDK device 0 from the public `rbln-smi` table. It used
neither privileged mode nor broad device visibility. The public table and
chosen mapping are recorded in the result before graph loading. Selecting
logical IDs from that table avoids assuming that a physical filename suffix
is the SDK's enumeration index.

All three failed attempts are retained:
[native-01](data/atom-binding-20261030/runs/atom-b1-native-01/result.json),
[native-02](data/atom-binding-20261030/runs/atom-b1-native-02/result.json), and
[native-03](data/atom-binding-20261030/runs/atom-b1-native-03/result.json).
They failed while creating the fresh runtime, reporting unavailable requested
device IDs 1, 3 and 2. Earlier harnesses passed physical filename IDs directly
to the SDK; native-02 also tried broader visibility. Native-03's public SMI
output explicitly showed physical `rbln2` as logical NPU 0. The corrected
parser handles that retained table in CPU tests, but the successful native-04
allocation was 0→0. **Successful native execution on a nonzero physical card
with remapped logical ID is not established.** The failures supply no passing
conformance or reload-latency samples. Their original result/Pod/API records,
Job specifications and cleanup receipts remain unchanged.

Artifact preparation also preserves its first failed attempt: the pushes
succeeded but writing a plan failed because the output directory was missing.
The [resumed preparation](data/atom-binding-20261030/preparation/evidence/atom-artifact-preparation.json)
retains those manifests and writes only the missing authentic CLI diff plans.
No synthetic manifest or hand-authored switch plan substitutes for them.

## Source identity and reproducibility

The runtime checkout was
`fbfdb8bb9aa86750f144fa4ead276fe17f6db537`, established by the
[controller freeze receipt](data/atom-binding-20261030/provenance/controller-freeze-atom-source-01.stdout)
and the [launch controller](data/atom-binding-20261030/provenance/native_job.py)
that checks that commit and mounts its source read-only. This commit identity
is controller evidence, separate from the application's recorded module hashes.
The graph producer was unchanged baseline
`f5fa13377a3d967ba3d4022fb73ae59b1c87441c`; its `atom.py` source hash matches
the native loader's source hash. The neutral fixture-capable kit was
`71c37d26f3ec9bedc20c93b27c2cfde9dfd64d53`; its packaged canonical-LF
[conformance source](data/atom-binding-20261030/harness/conformance.py) matches
the SHA recorded by the native harness.

[package-manifest.json](data/atom-binding-20261030/package-manifest.json)
records every packaged byte and the exact/canonical-LF source comparisons.
The adapter and shared CLI bytes match their native hashes exactly; inherited
Windows source files can differ by CRLF only. The [LOC snapshot](../docs/atom-binding-loc.json)
counts those complete source files separately from SDK implementations.
Raw exports, their gzip archives, per-file manifests, preparation scripts,
all four measured harness versions, the successful exact harness copy,
kit stdout/stderr and log-export manifest are retained with `-text` attributes.
The 15 GB graph and checkpoint payloads are not packaged. Artifact graph hashes
are cross-checked against producer and kit inventories; the six host-checkpoint
payload SHA-256 values are recorded by the native worker. Without those
payloads the local verifier cannot independently rehash them.

Recompute the evidence checks and [derived summary](data/atom-binding-20261030/derived-summary.json)
without Torch, a vendor SDK, hardware access, or a new native run:

```bash
python experiments/verify_atom_binding_evidence.py
```

The exact [native harness](data/atom-binding-20261030/harness/run_atom_b1.py)
can reproduce the protocol only with verified artifacts and an explicit
exclusive allocation. Executing it is a new hardware experiment. The current
record covers one short prompt, one graph/SDK stack and one successful card
assignment. Concurrency beyond owner-lock CPU tests, long contexts, changed
weight targets, B2 apply/rollback, device byte-space verification and whole-model
device digests remain outside this B1 result.
