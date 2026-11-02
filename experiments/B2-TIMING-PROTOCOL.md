# Current B2 latency protocol

Issue #45 measures the unchanged transactional RNGD binding, separately from the earlier direct row-update and shape-graph benchmarks. Runtime baseline: `4482faef87dabef8f586c6e7dff382b889cca115`; no `src/` changes.

Use the existing `native-interior-projection` provider case: base and interior checkpoint revision at B4/S256/group2. The authentic byte plan rounds changed rows outward to supported units. Inputs are the same four streams of eight fixed token IDs and eight generated IDs per stream as the retained independent fresh-target reference. This is a controlled changed-weight fixture, not a shape-graph switch or application prompt corpus.

One smoke pair precedes five main pairs. Each arm independently loads and generates the base outside the trigger clock. Pair order alternates B2/reload and reload/B2. The timed interval includes current owner planning, verified apply or explicit same-target reload (including closing the old owner), synchronization and native first-token time. Full continuation generation is a separate diagnostic. File delivery and external full-file hash validation precede the trigger in both arms. This protocol evaluates a prepared, local-file runtime transition, not Pod startup or network TTFT.

Current binding journaling and whole-parent readback verification remain enabled. Record logical serving writes and readback payload separately; neither is total DMA traffic. Do not sum overlapping runtime diagnostic durations. Require unchanged allocations/programs for B2, original compiled calls with native outputs, no compile attempts, base/historical-target token parity and B2 parity with each pair's current same-device fresh reload. Do not filter slow valid trials. Failed attempts remain separate and cannot enter a five-valid-trial summary silently.

Prepare and fully hash artifacts/cache off-card. `--check-config` uses only the standard library and does not allocate an owner:

```console
python -S bench/ccgrid_b2.py --config /absolute/config.json --check-config
```

Config fields: `source`, `cache`, `fixtures`, `references`, `input_manifest`, `source_manifest`, `source_commit`, `phase` (`smoke` or `measure`) and `run` (an output directory that must not yet exist). See the harness docstring for manifest formats. Full artifact hashes are supplied by the off-card preparation record; the native harness checks sizes/inventory plus small source/input hashes and records that distinction.

Execute only inside a bounded `cdx-` Kubernetes Job on the approved RNGD node with exactly one `furiosa.ai/rngd` resource, TTL and active deadline. SDK 2026.4.0 and Torch 2.10.0+cpu are required. Initialization records allocation UUID and physical/logical mapping; no device selection overrides the allocation. Existing model/cache mounts remain read-only. Preserve Job UID/status/logs and verify owned Job/Pod absence after collection.

Raw per-trial JSON and append/fsync events preserve incomplete attempts. The final result alone certifies the run-level original-call observer; immutable earlier trial snapshots deliberately retain their pending-native-gate status. Aggregate the final verified result and check those snapshots against it. Report median and inclusive IQR for each main arm with n=5; smoke is excluded by declared phase.
