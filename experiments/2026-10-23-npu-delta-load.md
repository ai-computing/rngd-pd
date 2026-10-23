# onggi-driven partial NPU weight loading

Date: 2026-10-23. Implementation: [PR #22](https://github.com/ai-computing/rngd-pd/pull/22).

## Scope

The earlier experiment verified RNGD partial updates using header-derived range
fixtures; it did not run onggi. ATOM then supported co-compiled selection and
independent-bundle preparation, without partial checkpoint updates.

This experiment exercises a real onggi store and CLI for both chips, including
ATOM partial weight updates through the upstream `DeviceWeightDecode` graph.
Native bucketing, `HostPool`, and optimum bundles remain separate ways to switch
embedded-weight ATOM models. The optimum implementation and its harness now
live in `atom_bundles.py` and `bench_atom_bundles.py`; existing imports from
`swap.atom` remain compatible.

## Protocol

- Model: meta-llama/Llama-3.1-8B-Instruct, revision
  `0e9e39f249a16976918f6564b8830bc894c89659` (32 layers).
- Fixture: perturb `model.layers.0.mlp.down_proj.weight`, shape
  `[4096, 14336]`, BF16, using `bench_rngd make-delta`, scale 0.1, seed 1.
  Both checkpoints are immutable. The changed tensor has 117,440,512 bytes.
- Publish materialized base/new checkpoint directories to an independent
  onggi store. onggi was actually built and executed, revision
  `b63d5022ae1586916d92fae37dcf9cdf51e58f77`, Go 1.27.3.
- The six checkpoint files total 16,060,581,181 bytes. The forward onggi diff
  marks one changed run of 117,595,127 bytes. Row refinement removes chunk
  boundary neighbours and leaves one changed tensor. The newly stored
  compressed payload is 78,681,918 bytes. Store push is outside the NPU timing.
- RNGD: one card, B4/S256, group 8, cached EDFs, `RNGD_MAX_SHAPES=8`.
  furiosa-torch 2026.2.0. ATOM: one RBLN-CA22, B1/S64, device-input weights, a single compiled callable.
- Python 3.10.12, PyTorch 2.11.0+cpu. ATOM: rebel-compiler 0.11.0,
  torch-rbln 0.2.2. Runtime source hashes and compiler/tool provenance are in
  `data/npu-delta-20261023/provenance.json`.
- Seed-0 random prompt IDs (`bench_rngd.prompts`), 8 prompt tokens, 8 generated
  tokens per stream. Fresh base and fresh destination run in separate processes.
  One resident engine performs new/base/new/base updates, querying onggi on
  each call. Generated tokens must equal the exact fresh destination and the
  fixture must produce different tokens from the base.
- ATOM also checks every unchanged device tensor keeps the same Python tensor
  object, and Dynamo's unique graph count stays fixed through updates. Its
  update runs a scratch step and synchronizes before committing staged inputs,
  because placement alone may defer the physical device upload.

`load_delta.total_s` includes the onggi invocation, planning, checkpoint
validation, weight conversion/upload, and (ATOM) the scratch execution.
`request_to_first_token_s` adds the full 8-token prompt's measured first-token
interval. Later generated tokens are excluded from this latency but compared
for correctness. Fresh-load time begins at worker entry and includes model
initialization, graph preparation, and that same prompt through its first token;
it excludes parent process-launch overhead. ATOM initial compilation/cache
preparation is included in the fresh first-token interval.

## Results

Seconds, with four bidirectional updates per backend:

| Backend | Fresh base → first token | Fresh new → first token | Live onggi + delta call | Delta + prompt → first token |
|---|---:|---:|---:|---:|
| RNGD | 27.43 | 35.74 | 1.117–1.998 | 1.499–2.568 |
| ATOM | 55.91 | 160.48 | 1.253–1.337 | 5.316–5.403 |

Each update transfers one **117,440,512-byte tensor**, rather than the full
16.1 GB checkpoint. This is the updater's logical weight payload; it excludes
scratch KV, activations and workspace and is not a hardware PCIe counter.
All **160 compared delta-generation tokens** equal their
fresh destination (128 on RNGD, 32 on ATOM). Both destinations differ from
base outputs, and each resident engine's initial output also equals the fresh
base. ATOM retains every unchanged input allocation and its Dynamo graph count
stays **1 → 1** on all four updates. Its scratch materialization takes
0.579–0.591 s and is included above.

Fresh timings vary with compilation/cache and system state. In particular,
ATOM's fresh-new first-token interval contains initial graph compilation; the
55.91 s fresh-base result benefited from cache preparation. These figures are
observed runs, not a promise of a fixed cold-start speedup. The retained path
reuses the already running graph, regardless of the fresh process's compile
cost. The preliminary RNGD run had 0.96–1.44 s update-through-first-token times;
the table uses the subsequent run without overlap with our ATOM experiments,
including its slower first update. No best-of-run selection is used.

Raw evidence: [RNGD](data/npu-delta-20261023/rngd.json),
[ATOM](data/npu-delta-20261023/atom.json),
[forward diff](data/npu-delta-20261023/onggi-forward.json),
[reverse diff](data/npu-delta-20261023/onggi-reverse.json),
[provenance](data/npu-delta-20261023/provenance.json).

## Reproduction

Use materialized checkpoint directories with matching safetensors layouts;
onggi does not follow source symlinks when publishing.

```bash
onggi push --store "$STORE" --ref base "$BASE"
onggi push --store "$STORE" --ref new "$NEW"
onggi diff --store "$STORE" --ranges --json base new > ranges-forward.json

RNGD_MAX_SHAPES=8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python experiments/verify_npu_delta.py --backend rngd \
  --base "$BASE" --new "$NEW" --store "$STORE" --onggi "$ONGGI" \
  --cache "$RNGD_CACHE" --batch 4 --sequence 256 --out rngd.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python experiments/verify_npu_delta.py --backend atom \
  --base "$BASE" --new "$NEW" --store "$STORE" --onggi "$ONGGI" \
  --cache "$ATOM_CACHE" --batch 1 --sequence 64 --out atom.json
```

Run on allocated hardware with the matching SDK environment. An ATOM Pod
requests one `rebellions.ai/ATOM`; compiles require a host C++ toolchain.
The final update APIs and constraints are documented in
[`docs/delta-swap.md`](../docs/delta-swap.md).

## Interpretation and limits

These are partial updates of resident weights. Initial loading still reads the
complete checkpoint. onggi refs describe safetensors, not byte offsets inside
compiled artifacts. Neither a `.rbln` diff nor identical file bytes provide a
mapping to NPU addresses. The update granularity is a complete changed tensor,
even when only a few rows differ; this implementation does not claim sub-row
device writes.

ATOM device-input graphs retain the upstream slower decode rate and initial
in-process graph preparation requirement on rebel 0.11. Embedded-weight graphs
retain their fast execution but require session/bundle reloads for new weights.
Updates need headroom for changed inputs and scratch KV, and run between
requests. Previous KV is invalid after a checkpoint or shape change.

The earlier [REA probe](2026-10-22-delta-swap.md) remains evidence of that
analysis's limits: it did not establish an external constant-buffer binding API.
This implementation uses public device tensor inputs and leaves vendor binaries
unchanged; it does not infer a private binding mechanism from the REA timeout.

An initial short natural-language prompt produced identical base/new outputs.
That run was not accepted as evidence of applying changed weights. The final
protocol uses discriminating random prompts and requires different destination
tokens as well as exact destination parity. An earlier RNGD run overlapped ATOM
initialization; isolated results are used for the final latency comparison.

## Validation and cleanup

CI passed 91 NPU-free tests on Linux. They cover actual CPU tensor transforms,
retention of unchanged inputs, failed transfer/probe preservation, RNGD rollback
and poisoned-state handling, old-KV rejection, live onggi command construction,
missing/error diff rejection, and the existing serving/native tests. The tier
scheduler test now uses an explicit first-step event before admitting its burst,
so its narrow-width assertion does not depend on thread scheduling.

Both ATOM test Pods completed with exit code 0. Only the experiment's own Pods
and namespace were deleted. RNGD benchmark processes exited after closing their
engines. Checkpoints, onggi store, compiled artifacts, raw logs and source clones
remain in the independent experiment directory for reproduction. The previous
provider session, its checkout and its resources were never resumed or changed.
