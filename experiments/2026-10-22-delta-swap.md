# Resident-weight delta loading on RNGD and ATOM

Date: 2026-10-22. Issue: [#17](https://github.com/ai-computing/rngd-pd/issues/17).
Implementation and review: [#22](https://github.com/ai-computing/rngd-pd/pull/22).

The optimum bundle implementation was subsequently moved to
`swap/atom_bundles.py` (with compatible imports), and its harness to
`swap/bench_atom_bundles.py`. Reproduction commands below use the current module
name. For real onggi execution and ATOM partial checkpoint updates, see the
[2026-10-23 follow-up](2026-10-23-npu-delta-load.md).

## Question and method

Can a running engine select another compiled shape without loading the model
weights again, and can changed RNGD weights be uploaded independently?

The baseline and swap measurements use the same engine implementation and
compiled artifacts. `fresh` starts a new process and uploads/loads the complete
model; `swap` retains one engine and changes its active variant. All latency
figures below include the first generated token for the stated prompt, rather
than timing a pointer assignment alone. Generated token IDs are compared with
the fresh load of the exact destination variant/checkpoint.

Code lineage: main `7858013`, initial RNGD WIP `ae9f8b5`, implementation
`4b493de`, warm-load instrumentation `08756df`. Each JSON record contains the
SHA256 hashes of the actual runtime and harness sources used. After those runs, upstream RNGD PR #21 and CI PR #20 were merged into this
branch; row refinement and Pod device selection were preserved. The earlier source
revisions differ in warm-load diagnostics, not the shared graph selection or
RNGD arithmetic. The prior provider session and its running resources were
only read; execution used an independent checkout and an independently
allocated ATOM Pod.

## Setup

| Item | Value |
|---|---|
| Host role | RNGD and ATOM node, shared with other workloads |
| CPU / memory | 2 x AMD EPYC 9254, 96 logical CPUs; 1,585,002,868 kB system memory |
| Kernel | 6.8.0-40-generic |
| RNGD | Furiosa RNGD, furiosa-torch 2026.2.0, driver 2026.4.0 (`32ffd4e`), PyTorch 2.11.0+cpu |
| RNGD model | meta-llama/Llama-3.1-8B-Instruct, revision `0e9e39f249a16976918f6564b8830bc894c89659`, BF16 |
| RNGD graphs | layer group 8; max unique shapes 8; B4/S256, B8/S256, B4/S512 |
| RNGD workload | 8 prompt tokens, 8 generated tokens per lane, seeded random IDs; OMP/MKL threads 4 |
| ATOM | RBLN-CA22, KMD 3.0.0; rebel-compiler 0.11.0, optimum-rbln 0.11.0.post1, PyTorch 2.11.0+cpu |
| ATOM shared bundle | KONI-Llama3.1-8B-Instruct-20241024, revision `0172bf7e73977ae213613529f3d2d5400a33f678`, B1/B8, context 512, shared compilation session |
| ATOM prompt | IDs `[128000, 791, 6864, 315, 9822, 374]`, 8 generated tokens per lane |
| ATOM warm-pool test | Qwen2.5-0.5B-Instruct, two independent compilations with prefill chunks 64 and 128, B1/S512 |

Artifacts and checkpoints were already on local storage; the OS page cache was
not flushed. One full-load baseline per variant, two RNGD swap rounds in each
of two runs, and three ATOM swap rounds were measured. These are engine
microbenchmarks, not HTTP load tests or production tail-latency guarantees.
The recorded mixed RNGD SDK/driver combination is the tested environment,
not a claim of vendor-supported compatibility for other versions.

## RNGD graph selection

| Variant | Full reload through first token, runs 1 / 2 | First selection of additional graph, runs 1 / 2 | Prepared revisit, runs 1 / 2 |
|---|---:|---:|---:|
| B4/S256 | 20.673 / 21.591 s | initial variant | 246.7 / 250.9 ms |
| B8/S256 | 20.578 / 22.949 s | 387.3 / 445.8 ms | 341.4 / 403.9 ms |
| B4/S512 | 22.655 / 23.472 s | 336.1 / 352.6 ms | 325.2 / 319.5 ms |

All **512 generated tokens** across the two runs' swap rows equal their fresh
destination variants. The prepared graph-selection call itself was about
0.01-0.02 ms; that is not the first-token latency above. Loading an additional
graph and its tables took 28-52 ms on the swap path.

Resident projection/LM-head tensor accounting remained **13.979 GiB**, with
one shared copy across all three variants. This is tensor-byte accounting,
not total HBM consumption: graph state, batch-shaped norms, KV and allocator
padding are additional. The benchmark does not fill HBM to estimate free space.

## RNGD changed checkpoint

The fixture perturbs only `model.layers.0.mlp.down_proj.weight`, with
`w * (1 + 0.1 * N(0,1))`, RNG seed 1, converted back to BF16. Checkpoint
directories are separate; the original tensors are not modified.

| Planning path | Tensors selected | Actual model bytes uploaded | Planning | Apply | Request through first token |
|---|---:|---:|---:|---:|---:|
| Local 1 MiB block comparison | 4 | 237,142,016 B | 2.505 s | 0.750 s | 4.133 s |
| Precomputed exact tensor range | 1 | 117,440,512 B | 0.007 s | 0.230 s | **0.947 s** |

A fresh load of the changed checkpoint took **22.430 s** in the second run.
Both update runs' generated tokens equal a fresh changed checkpoint and differ
from the base checkpoint. Coarse block comparison touches neighbouring tensors,
including a host embedding and batch-shaped norms; the planned full-tensor byte
count therefore differs from actual uploaded bytes.

The historical results above used `refine=False` (the then-current planner);
use `--no-refine` with the integrated harness to reproduce that boundary. The
current default refines flagged rows, preserving upstream behavior.

The precomputed range file was generated from this controlled fixture's
safetensors header. It uses the real `onggi diff --ranges --json` schema,
verified against `onggi/pkg/store/diff.go`; **onggi itself was not executed in
this measurement**. The range is conservative at whole-tensor granularity.

## Integrated refined planner verification

After merging upstream PR #21 and CI PR #20 at `03e57da`,
`verify_refined_delta.py` applied the fixture update and its inverse in three
rounds using the default `refine=True` planner. All **192 generated tokens**
matched the corresponding fresh destinations. Each update uploaded exactly
117,440,512 B. Request-through-first-token latency was **0.576–1.021 s** across
six updates (first forward update 1.021 s; later forward updates 0.576/0.610 s).
The source commit and runtime/harness hashes are in
[rngd-refined.json](data/delta-swap-20261022/rngd-refined.json).

Reproduce after producing the fixture and fresh-load reference above:

```bash
"$RNGD_PY" experiments/verify_refined_delta.py --base "$CHECKPOINT" \
  --new checkpoint_delta --ranges experiments/data/delta-swap-20261022/projection-ranges.json \
  --reference experiments/data/delta-swap-20261022/rngd.json --cache "$CACHE" \
  --chip 0 --rounds 3 --out refined.json
```

## ATOM shared compilation session

Full bundle reload through the first token: **20.343 s at B1**, **19.635 s at
B8**. With the bundle already loaded:

| Variant | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| B1/S512 | 83.3 ms | 71.8 ms | 71.2 ms |
| B8/S512 | 563.9 ms | 557.1 ms | 559.5 ms |

All **216 generated tokens** equal the corresponding fresh-load outputs.
SDK allocation reports show **15,011,414,016 B of Kernel weights** on the
prefill owner and no Kernel allocation on either decoder. The owner and both
decoder runtimes stay alive together; selecting a decoder does not reload
those weights.

## ATOM independent bundle preparation

For the two independently compiled 0.5B bundles:

| Measurement | Value |
|---|---:|
| Background preparation | 1.433 s |
| Prepared selection call | 0.025 ms |
| Prepared selection through first token | **10.03 ms** |
| SDK planned bytes, active bundle | 1,140,850,688 B (1.0625 GiB) |
| Additional SDK planned bytes, prepared bundle | 1,151,336,448 B (1.0723 GiB) |
| Kernel weights in each bundle | 989,855,744 B |

Tokens during preparation and after selecting the prepared bundle equal their
respective fresh loads. There is a material limitation: the request submitted
during preparation took **1.439 s to its first token**, and that token arrived
after preparation finished. The thread avoids the engine request lock, but this
run does not demonstrate uninterrupted native execution during loading. The
serialization mechanism was not isolated. Use the pool with this latency cost
in mind; background load and fast prepared selection are distinct operations.

This memory measurement is for 0.5B, not two 8B models. Two independent 8B
weight copies generally exceed a 16 GiB ATOM card; same-session compilation is
the measured 8B solution. Arbitrary partial replacement of embedded ATOM
weights is not implemented.

## REA investigation

[REA](https://github.com/morluto/rea) 6.7.0 was used for a static investigation
of `native_torch.cpython-310-x86_64-linux-gnu.so` (105,469,225 B, SHA256
`7449350b1a5d4f42b04a2e1416a26954a5bd9630bf78a6edc82af2398e873fda`).
No installed vendor binary was patched or executed by the static probes.

- REA's GNU objdump provider returned a stripped static symbol table and
  unchanged artifact hash.
- Full Ghidra 12.1.4 import did not complete within REA's 330 s startup limit.
- Binary-layout inspection required an explicit pwntools Python provider that
  was unavailable in that environment.

No complete native inventory, additional weight-binding API, or valid device
address mapping was established. This is not evidence that such a mechanism
cannot exist. The implemented path instead uses weight graph inputs and shared
compilation sessions, with hardware token-parity evidence above. A future REA
investigation should isolate a specific loader/weight-binding function rather
than import the entire stripped library without a bound.

## Bring-up failures retained in the record

- The first RNGD attempt uploaded weights before selecting eight-PE fusion
  and failed allocating 939,524,096 B. Initialization now selects fusion and
  mesh before any upload. After that selection, ordinal 1 had no allocator in
  this device view; successful measurements use ordinal 0.
- Loading the existing Torch-2.11 artifact tree with the venv's Torch 2.10
  failed unpickling its `.w` file (`state is not a dictionary`). Selecting the
  same Torch 2.11 tree used by the bake resolved it; no cached binary was
  modified to bypass the mismatch.
- The fresh ATOM container initially lacked global Python dependencies,
  timezone data and the Thunk library. Required host packages were mounted
  read-only and runtime libraries staged only into the independent validation
  workspace. Those failures produced no performance samples.
- Initial CI collection exposed missing `transformers` and `pyzmq` test
  dependencies. The workflow now installs both and passes on a clean runner.

## Reproduction and checks

```bash
export PYTHONPATH="$PWD/src:$TORCH_211_SITE"
export RNGD_MAX_SHAPES=8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1

"$RNGD_PY" -m rngd_pd.swap.bench_rngd make-delta \
  --ckpt "$CHECKPOINT" --out checkpoint_delta \
  --tensors model.layers.0.mlp.down_proj.weight --scale 0.1

"$RNGD_PY" -m rngd_pd.swap.bench_rngd all --ckpt "$CHECKPOINT" \
  --delta-new checkpoint_delta --delta-mode inplace --chip 0 --group 8 \
  --variants B4,S256 B8,S256 B4,S512 --P 8 --n 8 --mem --cache "$CACHE" \
  --ranges experiments/data/delta-swap-20261022/projection-ranges.json --no-refine --out rngd.json

python -m rngd_pd.swap.bench_atom_bundles --bundle "$ATOM_8B_BUNDLE" \
  --variants B1,S512 B8,S512 --tokens 8 --rounds 3 --out atom.json

python -m rngd_pd.swap.bench_atom_bundles --bundle "$ATOM_05B_C64" \
  --warm-bundle "$ATOM_05B_C128" --variants B1,S512 --tokens 8 --rounds 3 --out warm.json

PYTHONPATH=src python -m pytest -q tests/swap tests/native_v4 \
  tests/test_devices.py tests/test_continuous_backlog.py
```

`TORCH_211_SITE` and `RNGD_PY` must select the matching Torch/SDK combination
used to bake the EDF cache. The supplied projection range applies only to the
specified checkpoint revision and unchanged shard layout. Regenerate it from
`read_header()` and the checkpoint index for another layout; do not reuse an
offset from a compiled artifact. ATOM's device must be allocated by the device
plugin or otherwise accessible in its current RSD group.

CPU verification after integration: **69 passed, 1 platform skip on Windows;
70 passed in Linux CI** ([run](https://github.com/ai-computing/rngd-pd/actions/runs/39764757684)).
The tests cover rollback, incompatible checkpoints, stale KV, range validation,
lazy graph samples, shared owners, warm-load failure and selection policy.
Hardware benchmark processes completed; the independently allocated validation
Pod was removed after capturing the results. No previous provider resource was
resumed, stopped or changed.

## Evidence

- [Integrated refined RNGD verification](data/delta-swap-20261022/rngd-refined.json)
- [RNGD runs and source hashes](data/delta-swap-20261022/rngd.json)
- [ATOM shared-session run and source hashes](data/delta-swap-20261022/atom.json)
- [ATOM independent warm-pool run](data/delta-swap-20261022/atom-warm.json)
- [Fixture projection range](data/delta-swap-20261022/projection-ranges.json)
- [REA probe summary](data/delta-swap-20261022/rea.json)
