# Resident delivery evidence (issue #47)

The main matrix is **complete and verified**: r02-r06, four delivery arms each
n=5. `s01` is excluded native smoke; `s02` is the separate full-checkpoint
active-lease control. Additional capacity c01 uses independent fresh/resident
processes, P257/O8, n=1. Results and limits are documented in
`experiments/REPORT-RESIDENT-DELIVERY-20261102.md`; the derived summary is
`experiments/resident-delivery-summary-20261102.json`.

Every cohort retains the exact compressed worker export, its SHA/size manifest,
original trial configs/results, real CSI logs/RPC receipts, donor preparation,
mount/lease/inode witnesses, native counters and memory snapshots. Model
payloads and vendor binaries are absent. These are worker-recorded model hashes;
offline verification cannot independently rehash absent model files.

From the repository root:

```sh
python experiments/verify_resident_delivery.py --cohorts s01 s02
python experiments/verify_resident_delivery.py --cohorts r02 r03 r04 r05 r06 --summary
python experiments/verify_resident_native_package.py
```

The verifier checks exact archive/file bytes, pinned Git source fingerprints,
same-target native token parity, resident buffers, zero compilation, ownership
mechanisms and first-token time partitions. Mutation tests reject byte changes,
corrupt archives and semantic failures even after export hashes are recomputed.
It requires every frozen main cohort for the separate n=5 summary command in
`experiments/RESIDENT-DELIVERY-PROTOCOL.md`.

`provenance/` retains exact worker/controller stages, source identities, failed
preparations, resource and UID cleanup records. Later CPU-model preparation
metadata belongs to separate Onggi issue84, not this native matrix or its
native-job cleanup claim.

Full-bundle SHA validation is inside demand-to-first-token. Graph-copy is local
colocated file copying, not cloud transfer or a stock production FXB API.
Graph-only default-policy refusal remains a valid fallback observation. Donor
preparation costs are separately recorded, not free; component medians must not
be added to produce a total median. This matrix has zero inflight requests at
trigger and does not establish loaded-service drain or tail latency.
