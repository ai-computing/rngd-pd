# E8: native resident B/S switches, n=5 per condition

All 30 fresh-process samples completed and passed the same-target token gate.
The matrix covers three switch directions and two actual graph states. Every
sample retains all 113 shared projection/LM buffers and every graph module
loaded before the timed switch. Compilation attempts are zero; each process
performs exactly one full shared-weight upload during its initial engine load.

| Direction (group 2) | Target graph state | n | Switch→first token, s | Swap, ms |
|---|---|---:|---:|---:|
| B4/S256→B4/S512 | first-use | 5 | 0.271334 (0.011597) | 47.987024 (4.445187) |
| B4/S512→B8/S256 | first-use | 5 | 0.312644 (0.025376) | 46.637115 (15.987469) |
| B8/S256→B4/S256 | first-use | 5 | 0.227095 (0.006228) | 45.639501 (4.277040) |
| B4/S256→B4/S512 | prepared | 5 | 0.213432 (0.007923) | 0.089141 (0.012140) |
| B4/S512→B8/S256 | prepared | 5 | 0.245141 (0.012431) | 0.086521 (0.010638) |
| B8/S256→B4/S256 | prepared | 5 | 0.176546 (0.000401) | 0.088261 (0.014741) |

Values are median (IQR); quartiles use inclusive linear interpolation. The JSON
summary retains every original value, Q1, Q3, minimum and maximum in seconds.
The table converts only `swap_s` to milliseconds for readability.

The trigger starts immediately before `engine.swap(target)` after base native
generation, synchronization and identity snapshots. `swap_s` ends after the
swap and synchronization. Switch→first token uses the generation call start
plus the native engine-reported prompt TTFT; it is not a separately observed
streaming token-arrival timestamp. Instrumentation overhead is included.

In `first-use`, the target has not been loaded or executed in that process
before the trigger. Existing precompiled files are available in the graph
cache. In `prepared`, the process first loads and executes the target, returns
to the base, executes a base barrier generation and synchronizes. The timed
switch therefore reuses target modules already loaded and exercised in that
process. Its preparation cost is excluded from the switch interval and is
reported separately below. Engine construction, the initial full upload and
initial base setup also precede the switch trigger in both conditions.

| Direction (group 2) | Target graph state | Engine load, s | Excluded preparation, s |
|---|---|---:|---:|
| B4/S256→B4/S512 | first-use | 17.083347 (0.185304) | 0 (0) |
| B4/S512→B8/S256 | first-use | 17.159166 (0.690340) | 0 (0) |
| B8/S256→B4/S256 | first-use | 17.282258 (0.845221) | 0 (0) |
| B4/S256→B4/S512 | prepared | 17.398170 (0.354820) | 0.708879 (0.022395) |
| B4/S512→B8/S256 | prepared | 17.196062 (0.511136) | 0.798530 (0.010722) |
| B8/S256→B4/S256 | prepared | 17.151729 (0.202124) | 0.735646 (0.021081) |

The full shared upload corresponds to 15,009,316,864 logical payload bytes in
every process. `logical_shared_weight_upload_bytes` sums shared-group and LM
tensor payload sizes; it is not a measured physical DMA byte count. No second
shared upload occurs at the switch. Retained buffer records include object
identity, device pointer, shape, dtype, device and bytes. Graph module object
identities before and after are also retained. First-use processes record 510
native calls and 510 output-device checks (255 base + 255 target); prepared
processes record 1,020 of each (an additional 510 during preparation). Every
observed native output is checked for `furiosa:0`.

The environment continues the native conditions of [E6](E6-RESULTS.md):
Llama-3.1-8B, SDK 2026.4.0, Torch 2.10.0+cpu, group 2, host_workers 2,
intra-op 4 and physical RNGD card 0. Python is 3.10.12; exact package,
platform, invocation, environment and source metadata remain in every raw row.
Executed source is `bbceb1759ae0522c65921512f3dd2b8ca95b24dd`, clean at execution;
source-file hashes and the matrix adapter hash are recorded. The frozen plan
also records the checkpoint payload-manifest hash. The 30 serial subprocesses
share one bounded Kubernetes Job (no retry, 1,800-second deadline, 600-second
TTL), unlike E6's independent Jobs. Its submitted resource policy requests one
RNGD card, 4 CPUs and 64 GiB, with limits of one card, 12 CPUs and 128 GiB.

Each target's eight generated integer token IDs per lane match an independently
fresh-loaded reference of that same B/S variant exactly (32 IDs at B4, 64 at
B8). Base generation and, when present, target warmup and returned-base barrier
generation also match their references. Inputs contain the exact eight-token
prompts, expected tokens and original reference-source hashes. The three flat
input files are unchanged LF originals, so their public hashes equal the
executed reference hashes. The original references and superseded cross-shape
failure remain in [the reference evidence](references-and-cross-shape-observation.json).
B8 differs from B4 at generated token 4 of shared prompt lane 0; the other
three shared lanes have no observed divergence. That is a cross-shape token
observation, not a failed same-target switch. No dense-logit or activation
bitwise-equality claim follows from these token checks.

Evidence and reproduction:

- [Frozen trial order and conditions](e8-matrix-plan.json),
  [submitted Job](e8-matrix-job.json), [matrix result](e8-result.json),
  [operational record](e8-operational.json).
- [Raw-linked collection](e8-collection.json), [all statistics](e8-summary.json),
  [publication provenance and input hash map](e8-publication-index.json),
  [verified original export manifest](e8-export-manifest.json).
- [Controller cleanup](e8-cleanup.json) confirms the succeeded Job and Pod were
  removed after process exit. The [final actual API observation](e8-root-final-cleanup.json)
  records no owned Jobs, no owned Pods and no active RNGD requesters.
- From the repository root, reproduce the summary into a new output file:
  `python bench/ccgrid_summary.py experiments/data/ccgrid-npu-20261030/e8-collection.json --out NEW-E8-SUMMARY.json`.
  The CLI verifies every public raw SHA256 before aggregation. Collection
  metrics are exactly each row's `timings`, plus `engine_load_s`,
  `preparation_excluded_s` and `logical_shared_weight_upload_bytes`.

Publication validation ran `e8_controller.validate()` locally, rehashed all 63
exported small JSON/log files and the compressed export archive, checked the
three input hashes against the frozen plan, compared their tokens/source hashes
to the retained independent reference evidence, and ran both
`bench.ccgrid_summary.summarize()` and the documented CLI. All six groups have
five valid samples and no exclusions. Public rows preserve exact measurements,
tokens, counters and object identities; hostname, allocator UUID and private
paths are sanitized. Original private hashes and public hashes are distinct
and explicitly linked. Exported log hashes are retained in the manifest;
the original logs remain private. Native/checkpoint payloads were not downloaded
or independently rehashed during publication.

Scope: this matrix measures native resident switches using existing compiled
graphs. It contains no artifact delivery timing, delta/full delivery comparison,
HTTP gateway measurement or Pod→first-token result. OS page cache and background
node load are uncontrolled/shared. These measurements are separate from E6 and
are not pooled with its delivery or trigger intervals. Preparation and engine
load costs must accompany interpretation of the prepared switch timings.
