# Native ATOM parsed-model reuse

Issue [53](https://github.com/ai-computing/rngd-pd/issues/53), PR [54](https://github.com/ai-computing/rngd-pd/pull/54).
The merged common transition caller can reuse the public parsed host model on
real ATOM hardware. Five alternating pairs reduce trigger-to-first-token median
from **20.8113 s to 10.2899 s** (ratio of medians **2.0225**, **50.556%** less time).
All target continuations equal an independently loaded fresh target.

This result is restricted to an unchanged compiled graph and a publisher metadata
revision. It is not a shape change, changed weights, a B2 transaction, or retained
device weights. Both arms report B1 `reload` and create a new native runtime.

## Measured path and starting state

Runtime baseline `57e610eec6b91c5303060a3820241b0f4dd0c0c7`; immutable measured
harness/source `fec2d370cc35f66dcbbf6897bc1d41fed1b48850`, 82 Git-blob hashes.
Public `rebel` and compiler 0.11.0, Torch 2.11.0+cpu, KMD 3.0.0;
`etri-001`, one allocated RBLN-CA22/ATOM per Job (main `/dev/rbln2`, smoke
`/dev/rbln3`, each container SDK ordinal0).
Public `RBLNCompiledModel`, `create_runtime`, and `SyncRuntime.run` only.
No vendor internal API, compilation, driver change, cache dropping, or mock engine.

The existing authentic Llama-3.1-8B B1/S64 graph contains 32 layers and is
15,176,307,063 bytes, SHA256
`02b4eebd765c1acd3412517ea3804fe0ba38ddf2aa020a88db853245f2908ebe`.
Base and target graph paths are hardlinks to the same immutable graph inode;
publisher profile metadata differs. Artifacts and checkpoint were mounted read-only
and fully hashed before allocation and again after release. This experiment does
not include artifact delivery or a disk-cold parse.

Each arm starts with an independently loaded and generated base. Shared host
embedding/position Tables are already loaded, and the OS page cache is warm/uncontrolled.
A separately loaded fresh target generates the reference and closes before the
matrix. Prompt IDs `[1,42,256,1024,7,999,2048,13]`, eight greedy target IDs
`[220,16,13,16,13,16,13,16]`. Base load/generation and fresh-reference preparation
are outside the trigger and are retained in raw records.

Both arms call the same `AtomBinding.transition`: resident identity, plan parsing,
admission, reload and target identity postconditions. `full` supplies the same real
public loader as an explicit factory to disable parsed-object reuse; `reuse` uses
the default owner. Target parse/create counters are respectively **1/1** and **0/1**.
The old native runtime closes in both; only the reuse arm retains the host compiled
object identity. No device pointer-residency or transferred-byte claim is made.

The trigger starts immediately before `transition`. Its first-token endpoint is
generation-call entry plus the engine's measured prompt TTFT, checked against
native prompt-call durations and the enclosing generation clock. It includes the
full graph SHA, old runtime close, optional parse, new runtime creation and prompt
execution. This is a native caller endpoint, not HTTP readiness or Pod startup.

## Main results

Five matched pairs, full/reuse order reversed on odd pairs. An excluded one-pair
smoke precedes the main Job; no failed native attempt, retry or outlier exclusion.
All times are seconds; brackets show inclusive Q1–Q3. Medians of component times
need not sum to the median end-to-end time.

| Metric | Full parse | Parsed host reuse |
|---|---:|---:|
| Trigger → first token | 20.8113 [20.7909–22.4849] | 10.2899 [10.2814–10.7197] |
| IQR, trigger → first token | 1.6940 | 0.4383 |
| Transition call | 20.2470 | 9.7257 |
| Full admission/hash | 9.1642 | 9.1584 |
| Parse | 10.4790 | omitted; 0 native parse calls |
| New native runtime creation | 0.4126 | 0.3663 |
| Prompt TTFT after transition | 0.5656 | 0.5644 |

All first-token observations, in pair order:

- Full: 20.667264, 20.811289, 20.790898, 22.484855, 22.510782.
- Reuse: 10.085133, 10.281396, 10.289911, 10.719732, 11.002822.

Main has **315 completed native calls across 21 generations** (fresh plus ten base
and ten target generations), 16 parses, 21 runtime creations, zero compilation
attempts. All 168 generated IDs match the independent same-target reference.
Separate smoke has 75 native calls, four parses, five runtime creations; it is not
pooled into n5. Later observations are slower in both arms; uncontrolled host/cache
conditions are retained rather than filtered.

The finding is a host parsing saving. Whole-graph SHA remains the dominant reuse
cost. Native runtime creation has not been eliminated, so these results do not
establish an improvement to partial NPU device loading.

## Reproduction and evidence

[`data/atom-reuse-20261102`](data/atom-reuse-20261102) contains 60 sealed files plus
the seal: exact-byte compressed input/run exports, individual pairs, source archive,
pre/post hashes, SDK environment probe, Job/Pod/log/UID-cleanup evidence and launch
scripts. [`atom-reuse-summary-20261102.json`](atom-reuse-summary-20261102.json)
contains all observations, median, inclusive quartiles and IQR.

Each unprivileged `cdx-` native Job requests/limits exactly one ATOM, backoff0,
TTL600 and deadline900/1800. Smoke Job UID `227e99cd-d3f3-4698-8df3-ed9851e5b1a6`
and main UID `06d4d7dc-0f0a-42c1-9dcc-7a453b3580e9` were exported in terminal state
and deleted with UID preconditions. The final live API inventory has no accelerator
or `cdx-` Pods/Jobs. CPU preparation/probe/posthash request no accelerator.
The remote transport scripts require the research hub's Orca adapter and the
existing site SDK/fixtures; these environment dependencies are not bundled SDKs.

```sh
python experiments/verify_atom_reuse.py
python -m pytest -q tests/swap/test_atom_reuse_bench.py tests/swap/test_atom_reuse_evidence.py
```

The SDK-free verifier checks exported byte identity, every pinned Git source blob,
source archive, pre/post input equality, complete alternating matrices, native
types/counters, output equality, bounded one-card Pod identity, release receipts,
clock formulas and independent statistics. It verifies preserved evidence; it
does not rerun native inference. CI fetches full Git history for the pinned source.
Independent review found three acceptance gaps in the offline verifier; each was
reproduced by a failing regression, fixed and re-reviewed. Measured source/raw did
not change. Existing historical raw and the paper's NUMBERS ledger are untouched.
