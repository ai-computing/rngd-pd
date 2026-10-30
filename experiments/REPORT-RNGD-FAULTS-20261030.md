# RNGD native request drain and rollback failure, 2026-10-30

One bounded native Job passed the two compliance gates missing from the
[original B2 conformance run](REPORT-RNGD-BINDING-20261030.md), using unchanged
runtime `a2ebc3281df4173a886bc6a035f2483eaa360bf5` on physical RNGD 0 /
`furiosa:0`, SDK 2026.4.0, Torch 2.10.0+cpu and `B4-S256-g2`.
This is **n=1 compliance evidence**, with no timing or speedup claim.

| Gate | Recorded proof |
|---|---|
| Apply waits for a whole native request | Worker signaled before `owner.apply()`. It remained unfinished before and after the actual 255-call base generation while `owner.request()` was held; actual serving rows and base manifest stayed unchanged. Completion occurred after the release marker and the worker joined. See `drain` and `whole-request-held-base` in [result.json](data/binding-faults-20261030/runs/rngd-faults-native-01/result.json). |
| Apply and ordinary rollback still work after drain | Target/base token parity matches original fresh references; actual serving D2H row hashes restore the original base exactly. `drained_apply`, `normal_rollback`, `second_apply` and five `generations` retain buffer pointers/program identities. |
| A real partial rollback write fails closed | The injected write restored rows 125:159 through the original public-copy path, synchronized, and read the full unit back before throwing. Actual mixed SHA-256 `1ffe5def2bea62871411d114d7920a00629b4fc910ef0a3a06f1557baed3b687` equals the expected first-half-base/second-half-target bytes and differs from both complete states. `rollback_failure` records empty resident/no manifest, request refusal and `engine_closed`; that condition checks failed state, cleared serving tensors/programs and no quarantined owner. No generation ran after failure. |

The five generation gates contain **1,275 original compiled calls and 1,275
native output checks, zero compile attempts**. The measured partial restore
was 974,848 logical unit bytes. Eleven whole-parent readbacks accounted for
2,583,691,264 payload bytes; these are not DMA/PCIe measurements. The new run
reuses the committed [authentic interior plan](data/binding-20261030/preparation/inputs/interior.switchplan.json)
and [actual independent reference rows](data/binding-20261030/runs/rngd-b2-native-02/work/refsactual.json),
whose hashes match the recorded inputs. No checkpoint/model payload is copied.

The exact [measured harness](data/binding-faults-20261030/harness/run_rngd_faults.py)
has SHA-256 `3e4da7803b4b2db1c4a4f6b81eb4ce20db4cf99249244aea5e1c3ad4389664bd`,
matching the raw result and recovered [provenance record](data/binding-faults-20261030/harness/run_rngd_faults.provenance.json).
A later local UID-only edit was not the executed script and is not packaged.
Job/Pod UIDs are linked through the create/terminal records and
[cleanup receipt](data/binding-faults-20261030/runs/rngd-faults-native-01/cleanup.json),
which confirms both owned resources absent and exit code zero.

Raw JSON, the original compressed export archive, export manifest, Job spec
and measured harness are exact-byte copies under `data/binding-faults-20261030`,
marked `-text`. The verifier checks archive/extracted bytes, gates, timestamps,
input/reference hashes, cleanup UIDs, measured harness SHA and current runtime
source against native LF hashes (CRLF-to-LF canonicalization only):

```bash
python experiments/verify_rngd_fault_evidence.py
```

This follow-up changes evidence, documentation and CPU CI verification only;
runtime behavior and LOC snapshots are unchanged. It does not establish crash
recovery or service continuation after a fatal SDK/device error.
