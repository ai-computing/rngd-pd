# Resident delivery evidence (issue #47)

The main matrix is **running**, not complete. `s01` is the excluded five-arm
native smoke; `s02` is the separate full-checkpoint active-lease functional
control. `r02` is the first verified four-arm main cohort. Subsequent frozen
cohorts are r03-r06. No n=5 aggregate is published from this partial package.

Every cohort retains the exact compressed worker export, its SHA/size manifest,
original trial configs/results, real CSI logs/RPC receipts, donor preparation,
mount/lease/inode witnesses, native counters and memory snapshots. Model
payloads and vendor binaries are absent. These are worker-recorded model hashes;
offline verification cannot independently rehash absent model files.

From the repository root:

```sh
python experiments/verify_resident_delivery.py --cohorts s01 s02 r02
```

The verifier checks exact archive/file bytes, pinned Git source fingerprints,
same-target native token parity, resident buffers, zero compilation, ownership
mechanisms and first-token time partitions. Mutation tests reject byte changes,
corrupt archives and semantic failures even after export hashes are recomputed.
It requires every frozen main cohort for the separate n=5 summary command in
`experiments/RESIDENT-DELIVERY-PROTOCOL.md`.

Full-bundle SHA validation is inside demand-to-first-token. Graph-copy is local
colocated file copying, not cloud transfer or a stock production FXB API.
Graph-only default-policy refusal remains a valid fallback observation. Donor
preparation costs are separately recorded, not free; component medians must not
be added to produce a total median. This matrix has zero inflight requests at
trigger and does not establish loaded-service drain or tail latency.
