# E7: native weight updates and sequential worker restart

All 30 main samples pass both base and target same-checkpoint token gates. Four preceding smoke samples are retained and excluded from statistics. Six fixture/mode groups each contain five valid samples, with no retries or outlier removal.

| Fixture | Mode | n | Update→first token, s | Logical written bytes | Planning, s |
|---|---|---:|---:|---:|---:|
| lm | inplace | 5 | 0.631582 (0.006404) | 65536.000000 (0.000000) | 0.002558 (0.000021) |
| lm | replace | 5 | 1.126596 (0.001358) | 1050673152.000000 (0.000000) | 0.002536 (0.000055) |
| lm | restart | 5 | 20.359316 (0.562221) | 15011577856.000000 (0.000000) | 0.002567 (0.000043) |
| sparse | inplace | 5 | 0.634767 (0.005676) | 1835008.000000 (0.000000) | 0.004730 (0.000038) |
| sparse | replace | 5 | 0.963455 (0.033306) | 234881024.000000 (0.000000) | 0.004546 (0.000190) |
| sparse | restart | 5 | 19.947918 (1.317972) | 15011577856.000000 (0.000000) | 0.004648 (0.000153) |

Values are median (IQR), using inclusive linear interpolation. The summary retains every original value and every timing key without imputation. Timing fields differ by mode and are compared only inside their fixture/mode group.

The native clock starts at the synchronized, drained base-ready boundary. First token is estimated from the generation call start plus the engine-reported prompt TTFT; it is not an observed HTTP or streaming arrival. Planning (`plan_s`) runs before base loading and is excluded from the update interval. Resident setup and base generation also precede it. Restart includes old-worker close/exit and new-worker startup/import/load; the base worker has exited before target launch. Engine load fields in resident rows describe base loading, while restart rows carry target-worker load timings.

`logical_device_written_bytes` counts successful logical weight payloads, never physical DMA. Inplace includes the production density/fragmentation fallback and may incur extra strided staging copies. Replace rewrites the entire touched stacked GROUP tensor, including unchanged neighboring layers. Restart counts shared plus LM weights and per-batch weights/constants, excluding programs, KV, position tables and activations. The original `apply.uploaded_bytes` remains in resident raw records.

`plan.upload_bytes` sums full source-tensor sizes selected by the refined safetensors planner; it is not the logical bytes actually written. `plan.changed_bytes` counts candidate target-file bytes from input ranges, before row refinement. `plan.total_bytes` sums target tensor payloads. These quantities have different meanings and are not interchangeable with each other or DMA.

Both sparse and LM fixtures use Llama-3.1-8B, B4/S256/group 2, SDK 2026.4.0, Torch 2.10.0+cpu, host_workers=2, intra_op=4 and assigned physical card 0 / `furiosa:0`. Every resident sample records 510 native calls and output-device checks; restart records 255 per sequential worker. Compilation attempts and allocator retries are zero. Each base and target gate compares 32 exact integer token IDs against independently fresh-loaded references from the same checkpoint and variant; no dense-logit bitwise claim follows.

One serial measurement Job runs all 34 cases. The actual running-Job/Pod API observation records a 3600-second deadline, 600-second TTL and backoffLimit=0. Its sequential init and main containers each request/limit one RNGD, for one effective allocation. Main requests 4 CPUs and 64 GiB, with limits of 12 CPUs and 128 GiB. This running observation establishes policy; separate terminal and cleanup records establish completion and removal.

Executed clean benchmark source is `f0f08bfca4409b55226022760cacb789e97a458c`. Actual PR76 `onggi.switchplan.v1` output comes from source `2d84ac39374d71086436e65b967a4f718876a2f3`, binary SHA256 `47bfaae6bace003bc29c9223521a80b44526c25835dc36d350b2979585abc114`. Input bytes are unchanged LF originals, with their hashes checked against the frozen plan and CLI preparation record. Preparation records full SHA256 for every original fixture file; publication rehashes the exported small evidence locally, not the checkpoint/native payloads. The base and two target fresh-reference original hashes are linked explicitly.

This existing refined runtime performance path does not claim ADR-0010 B2. There is no actual device-readback SHA256 in artifact byte space or retained undo journal. See [the compatibility assessment](../../../docs/adr0010-rngd-fit.md). Background node load and OS page-cache state are shared/uncontrolled. This native-only measurement excludes artifact delivery and is not an HTTP or Pod-start clock.

Evidence and reproduction:

- [Frozen plan](e7-matrix-plan.json), [submitted Job](e7-matrix-job.json), [actual running-Job/Pod policy](e7-actual-job-policy.json), [result](e7-result.json), [operational record](e7-operational.json), [source provenance](e7-source-provenance.json), [470-file frozen source hashes](e7-frozen-source.json), [input preparation](e7-v1-preparation.json), [matrix adapter](e7/scripts/run_e7_matrix.py), [independent-reference adapter](e7/scripts/run_e7_references.py). Public adapter code is sanitized for review; its original executed SHA256 remains separate.
- [All statistics](e7-summary.json), [raw-linked collection](e7-collection.json), [publication index](e7-publication-index.json). The index links 30 main and 4 smoke raw records, two new raw target references, the retained base reference and all original/public SHA256 pairs.
- [Controller cleanup](e7-cleanup.json), [root final cleanup](e7-root-final-cleanup.json), [main export manifest](e7-export-manifest.json), [reference export manifest](e7-references-export-manifest.json).
- Reproduce from the repository root into a new file: `python bench/ccgrid_summary.py experiments/data/ccgrid-npu-20261030/e7-collection.json --out NEW-E7-SUMMARY.json`.

Publication verifies all 77 original exported JSON/log files across both exports and both compressed archive hashes, source/input/reference checks and cleanup. The documented CLI checks all public raw hashes before recomputing statistics. Public raw measurements, tokens, counters and source hashes are preserved; hostname, allocator UUID and private paths are sanitized. Original logs and archives remain private, with their hashes public.
