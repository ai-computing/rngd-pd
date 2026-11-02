# Current transactional RNGD B2: repeated update versus reload

The current, unchanged B2 owner completes the existing interior projection-row transition in **0.448449 s median [0.198801 s IQR]** from plan entry through the first native token, versus **13.783574 s [0.335147 s]** for same-target reload, **n=5 per arm**. The ratio of condition medians is 30.7; this is a prepared local-file runtime transition, not Pod startup or network TTFT. All ten target continuations match retained independent fresh references and their pair's current same-device reload. Smoke is separate.

| Main measurement | Current B2 | Same-target reload |
|---|---:|---:|
| Plan to first native token, median [IQR], s |0.448449 [0.198801]|13.783574 [0.335147]|
| Owner planning, median, s |0.000652|0.000630|
| Apply or close/reload wall time, median, s |0.292414|13.484161|
| Native prompt TTFT, median, s |0.156001|0.151524|
| Logical serving writes, B/trial |1,949,696|15,011,577,856|
| Whole-parent readback payload, B/trial |469,762,048|0|

Medians of component fields are not additive. The first-token endpoint is the per-trial generate-call entry plus the runtime's returned prompt interval, which excludes remaining continuation generation and pre-clock Python entry checks. Readback and journaling remain inside B2 apply. The logical write counter excludes temporary staging, D2H and other copies; neither byte row is measured DMA traffic. B2 readback wall-time median is0.193759s, a nested diagnostic rather than another phase to add. All five B2 times are retained:0.457683,0.244580,0.251755,0.450556,0.448449s. The spread is reported without assigning an unmeasured cause.

## Conditions and validity

Runtime baseline `4482faef87dabef8f586c6e7dff382b889cca115`; measured harness commit `6349df767832872fa9559c487579cea77358d4bf`, exact harness SHA256 `faf52c5dfa3d6a4bae231d5de2654e6a00ca7e43de128504e7c5f7bc7500b44c`. The existing provider's `native-interior-projection` case is a controlled checkpoint revision at B4/S256/group2; it is not a shape change or task-quality evaluation. The exported chunk plan rounds projection rows128:192 outward to125:193. Fixed synthetic inputs are four eight-token streams, each generating eight IDs. The target's32 output IDs differ from the base and equal fresh-target references.

Each arm independently loads and generates the same base outside the trigger. Both time current `owner.plan`; B2 calls normal `owner.apply`, retaining undo/readback, while reload calls explicit `owner.load(target)` and closes the base before replacement. Five pairs alternate arm order. Existing artifact and runnable-cache mounts are read-only; OS cache and unrelated host load are uncontrolled. Full artifact hashes were verified off-card against the prior native kit before allocation. The native harness checks sizes/inventories and small input/source hashes; offline reproduction cannot rehash omitted model/device payloads.

One RNGD on etri-001 was allocated only by bounded cdx- Kubernetes Jobs (one requested/limited RNGD, backoff0, TTL600s, smoke deadline900s/main1200s). Python3.10.12, Torch2.10.0+cpu and Furiosa2026.4.0; native package versions, image digest, CPU limits and physical0/logical0/UUID mappings are retained in raw records. B2 preserves serving allocation pointers and program object identities. Main observed5,100 original compiled calls and5,100 native-output checks, with zero compile attempts. Smoke observed1,020. Both Jobs and Pods were UID-cleaned; the final live cluster preflight has zero cdx or active accelerator Pods. The main Pod exited0/Succeeded before the Job controller's completion count updated; terminal Pod and UID cleanup are the authoritative completion gates.

## Evidence and reproduction

Raw package: [data/b2-timing-20261102/](data/b2-timing-20261102/), including all12 native trials (two smoke, ten main), append/fsync events, immutable pending-gate trial snapshots, final certified run records, input/config/source manifests, actual Job/create/status/cleanup receipts and global pre/post checks. There were no failed native attempts or excluded main trials. Operational preparation retries precede native measurement.

Windows `git archive` applied CRLF conversion to staged baseline source. The frozen source manifest's phrase “exact baseline Git blobs” is too strong: [source-reconciliation.json](data/b2-timing-20261102/control/source-reconciliation.json) preserves measured hashes and compares every one of80 canonical-LF files with its actual baseline Git object. Raw measurement/config/export hashes are always exact-byte comparisons. The benchmark harness itself was staged as LF. These distinctions are checked rather than silently rewriting evidence.

```console
python experiments/verify_b2_timing.py --root experiments/data/b2-timing-20261102
```

[Final verified summary](b2-timing-summary-20261102-final.json) reports n, median, inclusive Q1/Q3/IQR and every observation. The first [verifier attempt](b2-timing-summary-20261102.json) is retained as a failed analysis attempt caused by an over-strict Job-controller-count assertion; it is not a failed native trial. The intermediate `-verified.json` predates the added nested readback-time diagnostic. The corrected verifier checks matching terminal PodUID/exit/restart/cleanup and reproduces the final summary. Git objects are checked when available; source archives without `.git` explicitly record that Git-object proof is unavailable while comparing canonical source files and frozen hashes. CI fetches history and compares reproduced JSON with the committed final summary.

The earlier E7 row-update and E8 direct graph-switch measurements remain separate experiments. This result adds performance evidence for the checked B2 path; it does not expand supported tensor/layout cases or provide automatic CSI dispatch, cold-cache startup, arbitrary checkpoint compatibility, or cross-vendor resident updates.
