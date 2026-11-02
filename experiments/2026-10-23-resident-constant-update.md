# Resident compiled-constant updates

Date: 2026-10-23. Static REA evidence led to an opt-in ATOM constant refresh
engine. The full 8B device test below passed four alternating checkpoint updates
without replacing the native runtime or recompiling its program.

## REA investigation method

The requested [REAP article](https://discuss.pytorch.kr/t/reap-rea-55-rea/12255) describes investigation prompts, rather than a replacement native engine. We applied its feature-tracing and function-deep-dive method from [REA-Prompts](https://github.com/JoasASantos/REA-Prompts), pinned at `349a59ba476675889b6d88536fb42936ec02af83`: inspect one feature statically, retain addressed evidence, and distinguish facts, inference and unknowns.

[REA](https://github.com/morluto/rea) 6.7.0 and Node 24.11.1 were installed into a new isolated analysis directory. Its GNU objdump integration inspected the small ATOM thunk and Python binding, and selected functions in runtime/compiler libraries. This produces assembly, not decompiled pseudocode. It does not repeat the earlier whole-library Ghidra import of the RNGD `native_torch` binary.

REA 6.7.0 lacks the `--symbol` option described on repository main. For large libraries, transparent shell wrappers appended GNU objdump's exact `--disassemble=<symbol>` selector. Each wrapper's source rule, hash and selected symbol are recorded in [rea.json](data/resident-update-20261023/rea.json). The smaller thunk and binding were disassembled once without a wrapper. All 13 observations completed without truncation, with exit code zero and unchanged before/after artifact hashes. No installed binary was patched, existing provider session resumed, or agent configuration changed.

## REA confirmed native behavior

The evidence ledger contains exact binary hashes, function addresses and REA Evidence IDs. Its principal findings are:

| Finding | Addressed evidence | Consequence |
| --- | --- | --- |
| The Python `apply_extern_transforms` entry calls `RuntimeBase::ApplyWeightTransforms` with skip-already-applied set to true. | `librbln.so`, `0x62b8415` through `0x62b842c`; `ev_48701dee71f868240f2b64b1e1bf1752111ca799aa0adc2b3ecb841c5e1b26fa` | Changed host data alone can leave an already-applied graph skipped. |
| `reset_applied_transform_graphs` clears graph bookkeeping without clearing constants. | Binding registration `0x71ee0`; function `0x75220` through `0x75286`; `ev_ed05eec8369ba8cfbff165fe61236e3a5bf0f6ddef7e5667dc3e36424a442e03` | Reset this bookkeeping before reapplying a changed group. |
| `BuildTransformRuns` checks applied graphs and supplied dependency keys before creating jobs. | `librbln_runtime.so`, `0xfc178d` onward; `ev_31e2557c9ca520366a902c45d34b2e47c1191e91ccf853a2c098d90ee3a00502` | Select complete transform dependency groups. |
| `ExecuteTransformRuns` serializes transformed weights and copies prepared runs from host to device. | `librbln.so`, `0xa95c0c3`, `0xa95c728`; `ev_5ce096452aece0dd56e19a8022b9aad7138d69850c130a001c444af4fe5cff78` | This is a concrete native update path; it does not establish a generic raw-byte patch API. |
| Host parameters and IO patch batching have distinct entry points. | `ApplyHostParamsOnly` at `0x62b8450`; begin/end Evidence in the ledger. | Apply host parameters only where needed. IO pointer patching is a separate operation. |

The object layout agrees across the binding and runtime observations. The Python object embeds `RuntimeBase` at `+0x40`; reset clears Python offsets `+0xd8` through `+0xf0`, matching runtime applied-graph bookkeeping at `+0x98` through `+0xb0`.

The pinned 3.0 thunk also exports `rblnCreateConstantBufferPool`, `rblnAllocateConstantBuffer` and `rblnSetConstantBufferData`. Its data-setting path writes into an existing pool after checking indices and readiness. A supported external runtime handle or arbitrary byte-offset signature was not established, so these exports are not an implementation recommendation.

## REA candidate sequence and limitations

The static evidence supports testing this sequence: retain a weight-free compiled runtime; map changed tensor names to meta-load keys; expand them to complete `extern_transform_key_groups`; reset applied-graph bookkeeping; and apply only selected complete groups using contiguous CPU tensors. Apply host parameters separately when required.

The active SDK's protected `core/weight_free.py` and readable `core_ori/weight_free.py` are different files. Their respective SHA-256 values are `19d0e17d95ac3a3c62019082d497fc3d7f57bb5f37455d77c20440dad35dffe8` and `71e3f2741c5000c831e43ddf4c7e8324115295a493ca5f6d1667845491812b48`. Their equivalence was not assumed: the companion API investigation confirmed methods on the active class, while REA established native implementations and skip/reset behavior.

Static assembly alone does not measure upload volume or speed, establish concurrent update safety, or prove full-model weight-free compilation succeeds. These REA observations establish no new RNGD binding or hardware-performance result.

## Metadata emission gate

The active SDK defaults to weight-free mode `off`. `RBLN_WEIGHT_FREE=1` at
process launch normalizes to `on`. Active frontend source additionally requires
weight sharing and nonscalar attribute constants. Direct compilation of the
original module supplies those constants; `_intmd_mode="gen"` alone does not
enable the feature. [Gate evidence](data/resident-update-20261023/rea-emission-gate.json)
and [flag evidence](data/resident-update-20261023/rea-weight-free-flag.json)
preserve the exact source and native observations.

## Bounded real-device proof

An allocated RBLN-CA22 ran a stack of sixteen 4096×4096 FP16 matrices: 512 MiB
of logical weights. The B1 program owned sixteen singleton transform groups;
the B4 program reused their constants and introduced no new weight-transform
groups. One changed 32 MiB group was applied through the native runtime.
Both B1 and B4 produced changed outputs matching the CPU destination reference.

Across two recorded runs, forward native apply-call time was 0.228/0.419 s,
and steady B1/B4 execution medians were 2.38–2.60 ms. The production updater
also performed the reverse change: both programs then matched their original
NPU output arrays exactly (`rtol=0`, `atol=0`). The reverse updater staged
33,554,432 bytes and one transform group. These are logical transform-input
bytes, not measured bus traffic. [All samples and source/result hashes](data/resident-update-20261023/toy.json)
are retained.

The first device-input layout experiment encountered compiler errors and CPU
fallback, so its apparent success and timings were rejected. Subsequent probes
disabled fallback. Fast embedded controls with weight-free mode off ran on the
NPU but exposed no update metadata; their partial-update assertions failed.
Neither negative control is reported as a partial-update success.

## Full Llama-3.1-8B update

Frozen implementation `ecfc6c3daa79fc76b9b077e53718dd7ee9e2defb` ran on an
allocated RBLN-CA22 with rebel 0.11.0, torch-rbln 0.2.2 and Torch 2.11.0+cpu.
The worker used `RBLN_WEIGHT_FREE=1`, `TORCH_RBLN_DISABLE_FALLBACK=all`, four
OMP/MKL threads, a 16-CPU limit and a 128 GiB RAM limit. Checkpoints/store were
immutable read-only inputs. The harness used B1/S64, a seed-0 random eight-token
prompt and eight greedy output tokens. One BF16 `down_proj` tensor of shape
`[4096, 14336]` differed; initial compilation and updates both convert it to FP16.

Fresh base and destination references ran in separate processes. A third
process retained its native runtime for `new → base → new → base` updates through
the real `onggi diff --ranges --json` and tensor planner. Each update changed
exactly one transform group with 117,440,512 logical input bytes (112 MiB),
about 0.73% of the checkpoint tensor payload. It did not rebuild the complete
checkpoint or replace the compiled program. All 32 compared generated tokens
matched their corresponding fresh reference exactly; the two references
produced different tokens, so a no-op update could not pass.

| Round / target | onggi + planning (s) | Device-update call (s) | Complete delta call (s) | Delta through first token (s) | Decode step (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / new | 0.805 | 1.278 | 2.104 | 2.659 | 68.747 |
| 0 / base | 0.797 | 1.315 | 2.147 | 2.700 | 68.715 |
| 1 / new | 0.811 | 1.325 | 2.163 | 2.737 | 68.712 |
| 1 / base | 0.796 | 1.311 | 2.143 | 2.697 | 68.707 |

Initial compilation/loading took 168.412 s for the fresh base, 154.642 s for
the fresh destination, and 151.169 s for the resident worker. Those costs are
outside the update rows. The delta-through-first-token column adds the measured
synchronous prompt TTFT (0.552–0.575 s) to the complete delta call; it excludes
generation of the remaining seven tokens. It includes planning, conversion,
native application and any deferred work observed by that first response.
It is not an HTTP/gateway measurement or a comparison against a precompiled
ordinary bundle's cold load.

All four rows retained the identical native runtime object and one compiled
program. Dynamo graph counters stayed zero because this backend uses direct
compilation; nonzero native program counts, actual device allocation and the
fallback-disabled execution establish the NPU path. The LM head is compiled
on the NPU; KV is host-resident. The unit of update is a complete affected
tensor/transform group, not individual changed bytes inside that tensor.

[All original timing samples, token IDs, exact implementation/native-library
hashes and configuration](data/resident-update-20261023/llama8b.json) are retained
with the original result SHA-256. Logical transform-input bytes are not measured
PCIe/DMA traffic. Four updates on one short prompt are a correctness and latency
observation, not a concurrency, long-context or production-serving claim.

## Comparison with the existing device-input path

After the constant engine exited, the same allocated card and Pod ran the
existing `DeviceWeightDecode` from the same frozen revision. SDK, checkpoint,
B1/S64, prompt, output length, OMP/MKL thread counts and resource limits matched.
Its process used the normal `RBLN_WEIGHT_FREE=0` mode, verified by the active
SDK getter, and `TORCH_RBLN_DISABLE_FALLBACK=all`. One warmup compiled its graph;
three subsequent generations retained one nonzero Dynamo graph and identical
tokens. Warmup is recorded separately and excluded from steady timings.

| Execution path | Measured steady decode-step medians (ms) | Prompt through first token (s) |
| --- | ---: | ---: |
| Existing device-input path, three repeated requests | 510.488 / 509.024 / 515.454 | 4.106 / 4.087 / 4.096 |
| Constant path, four requests after updates | 68.747 / 68.715 / 68.712 / 68.707 | 0.555 / 0.552 / 0.575 / 0.554 |

The ratio of the two median decode-step measurements is about 7.4×. This is
an execution-path comparison: graph implementation/fusion and host versus device
KV handling differ, so the ratio cannot be attributed solely to weight storage
or packing. Both compile their LM head on the NPU. This control measures
generation, not an old-path delta operation; it does not establish a 7.4×
improvement in native update-call time. The constant path's update call itself
still took 1.278–1.325 s, plus the separately reported planning/first-response
costs. Shared-host background load and the short, single-prompt sample limit
generalization. [Control samples and source provenance](data/resident-update-20261023/device-input-control.json)
preserve the warmup and every measured repeat.

## Provenance and resource cleanup

[Provenance](data/resident-update-20261023/provenance.json) records immutable
fixture manifest IDs, Onggi revision/binary hash, the container image, SDK/native
library hashes, CPU/RAM limits, background load and exact raw-export hashes.
The 23 exported worker, configuration and sampler files match their remote
byte hashes. KMD 3.0.0 was captured; the assigned card's firmware version was
not captured and is left unrecorded.

Joining native-context PIDs to the same sample's owned Pod cgroup observed all
three constant-path worker contexts and the control's context. Their maximum
rounded driver allocation labels were 14.2 GiB and 14.9 GiB respectively;
these are not byte-exact physical peaks. Foreign process contexts were excluded.

The owned Pod exited normally with code zero. Pod and namespace deletion used
UID preconditions; absence was confirmed at 06:16:48 UTC, followed by an audit
with no owned processes or device contexts. Source, checkpoint views, caches and
raw evidence remain for reproduction. The 485,060,608-byte full-matrix cache
and 178,978,816-byte control cache are separate from earlier benchmark stores;
the read-only fixture hardlinks did not create another physical checkpoint copy.

## Implementation validation

The opt-in `ConstantWeightDecode` uses direct gen-mode compilation and the
existing fast batched Llama graph, with host KV. `ConstantWeightUpdater` expands
changed keys to full transform dependencies, counts every native eligible
group, retains host-parameter pointer owners, stages independent replacements,
and rolls back failed/interrupted updates. A failed rollback or interruption
between device application and engine-state commit disables further inference.

Independent review's two Important findings (overlapping groups and interrupted
transactions) were reproduced RED and fixed GREEN. Additional regression tests
cover aliased replacements, host pointer lifetime, explicit compile examples,
packed `o_proj`, checkpoint parity and KV invalidation. The full local suite
passed 279 tests, skipped 6 environment-dependent tests, and passed 4 subtests.
[Fresh Linux CI](https://github.com/ai-computing/rngd-pd/actions/runs/39908041073)
passed 283 tests, skipped 2, and passed 4 subtests. The fixed two-layer Llama
smoke used one native program and measured a 23.96 ms decode step; the full 8B
test above then passed on the same frozen implementation.
