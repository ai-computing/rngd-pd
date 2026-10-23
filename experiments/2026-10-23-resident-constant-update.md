# Resident compiled-constant updates

Date: 2026-10-23. The REA sections below record static ATOM binding evidence. Device results are recorded separately when available.

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

Reusing one runtime while preserving unrelated resident constants remains an inference until a device test demonstrates changed outputs, stable runtime identity and allocation behavior. Static assembly does not measure upload volume or speed, establish concurrent update safety, or prove full-model weight-free compilation succeeds. These REA observations establish no new RNGD binding or hardware-performance result.
