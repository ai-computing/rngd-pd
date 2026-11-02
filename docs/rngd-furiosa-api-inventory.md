# RNGD binding: function-level Furiosa API inventory

Audit date: 2026-10-31. Source baseline:
[`be468b678957e6adb5e7438c5d263465a280a05e`](https://github.com/ai-computing/rngd-pd/tree/be468b678957e6adb5e7438c5d263465a280a05e),
`integration/onggi-prepared-serving`. Requested in [issue #43](https://github.com/ai-computing/rngd-pd/issues/43).
This is a source/documentation audit for paper review, with no licensing,
redistribution, ownership, or other legal conclusion. No SDK package internals
were inspected and no hardware or remote node was used for this audit.

The adapter adds no direct Furiosa import. The production owner nevertheless
depends on the inherited engine's SDK cache reconstruction and synchronization.
`allow_compile=False` prevents cache-miss compilation; it does not eliminate
SDK imports, compiler enum lookup for cache keys, or native deserialization.

## Source and version boundary

The source scope is the 18 explicit `full_rngd_runtime` files in
[`rngd-binding-loc.json`](rngd-binding-loc.json): 3,343 physical and 2,938
nonblank lines, including dormant functions. These are whole-file counts,
not counts of private APIs or executed statements. The adapter's two files
contribute 634 physical / 567 nonblank lines. SDK implementations, tests,
artifacts and documentation are excluded. See [the scope explanation](rngd-binding.md#reproducible-loc-scope).

The [existing native evidence](../experiments/REPORT-RNGD-BINDING-20261030.md)
records one bounded B2 pass with SDK 2026.4.0, Torch 2.10.0+cpu,
`furiosa:0`, `B4-S256-g2`. This audit does not rerun or expand that evidence.
`fsdk.NEW` selects the new branch using the installed **furiosa-torch package**
version, threshold `(2026, 3)`; the fallback version is `unknown`, which selects
the legacy branch ([`fsdk.py:20–26`](../src/rngd_pd/engines/fsdk.py#L20-L26)).
Do not substitute an umbrella SDK version for this package test: the official
[2026.3 release component table](https://developer.furiosa.ai/v2026.4.0/en/whatsnew/release-2026.3.0.html)
lists furiosa-torch 2026.2.0. Branch/version comments in this repository are
source observations, not a vendor compatibility guarantee.

Classification used below:

- **Explicit private**: underscore SDK namespace/member used by the source.
- **Native/undocumented here**: exported SDK name or native binding whose exact
  contract was not verified in the public sources reviewed. A non-underscore
  name alone does not establish a supported public ABI.
- **Legacy**: retained older SDK branch, not validated by this audit.
- **Repository-private**: our own engine state and methods, not vendor internals.
- **Public Torch surface**: ordinary Torch calls; backend implementation still
  requires the matching Furiosa plugin and separate device validation.

## No-compile B2 reachability

[`RngdBinding.load():154–181`](../src/rngd_pd/swap/rngd_binding.py#L154-L181)
constructs `SwappableDecode(..., allow_compile=False, host_workers=2)` and calls
`engine.swap()`. `SwappableDecode.__init__()` imports `fsdk`, configures the
device and uploads weights. `swap()` calls `prepare()` → `_graphs()` → both
graph-spec functions → `artifact_cache.load()` for cache hits. A miss raises
`FileNotFoundError` ([`rngd.py:267–287`](../src/rngd_pd/swap/rngd.py#L267-L287)).

Importing `engines.decode_batch` also imports `kernels.decode_batch`, which
calls `_open_device(8)` at module scope and imports `kernels.dsl_batch`, which
calls `_open_device(8, 2)` at module scope. These side effects remain reachable
without calling any graph builder. `lmhead_graph_spec()` selects tactic enums
and uses their string representation in the cache key before returning its
uncalled `build` closure.

Phase labels: **I** import; **L** load/prepare; **S** serving; **U** apply,
rollback and release synchronization; **C** cache construction/compile only;
**D** dormant helper/CLI in the counted source. Reachability is static for the
default production factory; injected CPU factories do not exercise this path.

## Vendor dependencies and replacement status

Every source link below resolves against the pinned baseline as well as this
documentation-only branch. **V1–V6** are concrete vendor requests below.
“No verified public equivalent” means the reviewed sources did not establish
one; it does not mean that no such API exists.

| ID | SDK module / symbol or record | Repository caller and exact lines | Purpose, phase, no-compile B2 reachability | Classification; public replacement status |
|---|---|---|---|---|
| F01 | `furiosa.torch` registration/import | [`fsdk` module:16–18](../src/rngd_pd/engines/fsdk.py#L16-L18); [`kernels.decode_batch` module:33–40](../src/rngd_pd/kernels/decode_batch.py#L33-L40); [`dsl_batch` module:9–16](../src/rngd_pd/kernels/dsl_batch.py#L9-L16) | I, reachable during production load; loads the Torch backend before device use. | Vendor plugin dependency; no claim that importing it makes all exported names public. Public high-level deployment alternative P1 below; not a drop-in. |
| F02 | `furiosa.torch.set_fusion`, `set_mesh_kind`, `MeshKind.Mesh` | [`fsdk.open_device():130–137`](../src/rngd_pd/engines/fsdk.py#L130-L137), imports at [31–32](../src/rngd_pd/engines/fsdk.py#L31-L32); [`SwappableDecode.__init__():148–177`](../src/rngd_pd/swap/rngd.py#L148-L177); module calls [`kernels.decode_batch:34–36`](../src/rngd_pd/kernels/decode_batch.py#L34-L36), [`dsl_batch:10–11`](../src/rngd_pd/kernels/dsl_batch.py#L10-L11) | I/L, reachable. Configures fused/meshed PE address space before uploading tensors. | Top-level exports; exact supported contract unverified. P2 is a documented higher-level configuration alternative, not verified equivalent mapping; V4 needed. |
| F03 | `furiosa.torch.current_stream(...).synchronize()` | [`fsdk.sync():113–127`](../src/rngd_pd/engines/fsdk.py#L113-L127); [`SwappableDecode._sync():194–196`](../src/rngd_pd/swap/rngd.py#L194-L196); [`RngdBinding.apply():351–381`](../src/rngd_pd/swap/rngd_binding.py#L351-L381), [`rollback():390–410`](../src/rngd_pd/swap/rngd_binding.py#L390-L410), [`SwappableDecode.close():618–626`](../src/rngd_pd/swap/rngd.py#L618-L626) | L/U, reachable. Completion barrier before/after mutation and before release; legacy `sync()` is a no-op. | Top-level stream API and stream method; public contract unverified. No verified public equivalent preserving these transaction barriers; V4 needed. |
| F04 | `furiosa.torch._C.config.compiler.{CompilerConfig,TacticHintConfig,TacticSortingPolicy}` | [`fsdk` module:75–80](../src/rngd_pd/engines/fsdk.py#L75-L80) | I, conditional and reachable: executed when the first `furiosa.torch.compiler` import raises `ImportError`, even with compilation disabled. | **Explicit private namespace**. Repository first tries the non-underscore `compiler` path; its existence/support on the measured package is not established by this audit. No verified public equivalent for fallback on that package; V3 needed. |
| F05 | `furiosa.torch.compiler.Config`, tactic enums (or F04 aliases); `TacticHintConfig.ForLlmModelComputeIOBalanced`, legacy `ForLlmModelDecode`/`NoConstraint` | [`fsdk.hint():87–101`](../src/rngd_pd/engines/fsdk.py#L87-L101); [`decode_graph_spec.build():69–71`](../src/rngd_pd/engines/decode_batch.py#L69-L71); [`lmhead_graph_spec():143–158`](../src/rngd_pd/engines/decode_batch.py#L143-L158) | L: LM enum lookup and stringification for cache key reachable. C: config construction, decode hint alias lookup, and compile calls not reached by B2. `TacticSortingPolicy` is imported but unused in the scoped files. | Native/config coupling, private when supplied by F04. P3 exposes higher-level compile controls, without a verified equivalent to tactic aliases or `max_num_unique_shapes`; V3 needed. |
| F06 | `furiosa.torch.TclModule` via repository `TacticKernelModule(dsl)`; legacy `custom_ops.TacticKernelModule` | [`fsdk.TacticKernelModule():37–51`](../src/rngd_pd/engines/fsdk.py#L37-L51); [`DecodeLayer.__init__():59–67`](../src/rngd_pd/kernels/decode_batch.py#L59-L67); [`lmhead_graph_spec.LmHead.__init__():123–126`](../src/rngd_pd/engines/decode_batch.py#L123-L126) | C only for graph constructors. `_symbolic_axes` strips inline axis sizes before passing DSL to the new SDK. Imports remain I-reachable. | Version-specific graph/DSL API; not explicitly private by spelling. Public TCL/FXB direction exists (P3), but no verified drop-in preserving this DSL or exported graphs; V5 needed. |
| F07 | `furiosa.torch.CompileModule.from_module(..., compiler_config=...)` | [`decode_graph_spec.build():56–71`](../src/rngd_pd/engines/decode_batch.py#L56-L71), [`lmhead_graph_spec.build():152–158`](../src/rngd_pd/engines/decode_batch.py#L152-L158) | C only: builders are created but not invoked with B2's `allow_compile=False`. | Top-level graph compiler surface; supported public contract unverified. P3/P4 provide other compilation paths, not verified drop-ins for these graph input/output layouts; V5 needed. |
| F08 | `CompileModule.runnable._inner.serialize()` | [`fsdk.runnable_blob():55–58`](../src/rngd_pd/engines/fsdk.py#L55-L58), called by [`artifact_cache._save_runnable():75–80`](../src/rngd_pd/engines/artifact_cache.py#L75-L80) through [`save():104–108`](../src/rngd_pd/engines/artifact_cache.py#L104-L108) | C only; serializes compiled program into the repository's `.runnable` archive. Not called by no-compile B2 load/apply. | **Explicit private member `_inner`**. No verified public equivalent for serializing this Python module. P1 is a different artifact/runtime path; V1 needed. |
| F09 | `furiosa.torch.ir.Runnable.deserialize(blob)` | [`fsdk.module_from_blob():60–64`](../src/rngd_pd/engines/fsdk.py#L60-L64), called by [`artifact_cache.load():119–133`](../src/rngd_pd/engines/artifact_cache.py#L119-L133) | L, reachable on every new-SDK graph cache load. Native opaque bytes become a native runnable. | Native IR binding, no underscore in the used name; not equated with explicit private naming. No verified public equivalent for this cache format; V1 needed. |
| F10 | `furiosa.torch.Runnable(native)`, positional `CompileModule(runnable, module_weight, output_reconstructor)` | [`fsdk.module_from_blob():61–64`](../src/rngd_pd/engines/fsdk.py#L61-L64) | L, reachable. Reassembles the Python callable wrapper and state around deserialized program. | Undocumented-here wrapper constructor/record ABI. No verified public equivalent preserving the current archive and callable signature; V1/V2 needed. |
| F11 | `module.module_weight`, `module.output_reconstructor` | [`artifact_cache._save_runnable():78–79`](../src/rngd_pd/engines/artifact_cache.py#L78-L79); saved record consumed by [`fsdk.module_from_blob():63–64`](../src/rngd_pd/engines/fsdk.py#L63-L64); legacy weight save [`artifact_cache.save():114`](../src/rngd_pd/engines/artifact_cache.py#L114) | C for module attribute reads; L for trusted `state.pt` unpickle and constructor arguments. Both state fields remain required by no-compile reconstruction. | SDK record-layout coupling; fields are not underscore-private by spelling. `torch.save/load` is public Torch serialization, but does not establish a public SDK record schema. No verified public equivalent; V2 needed. |
| F12 | Reconstructor `zero_sized_tensor_info.output_indices_and_tensors`, `passthrough_pairs[*].target_tensor` | [`artifact_cache._output_metadata():56–72`](../src/rngd_pd/engines/artifact_cache.py#L56-L72), called at [79](../src/rngd_pd/engines/artifact_cache.py#L79) | C only for inspecting/replacing record fields. Removes export-time FakeTensors using `dataclasses.replace` and Torch meta tensors. L restores the resulting record through F11; does not call this helper. | SDK record/dataclass schema dependence; no verified public equivalent or guarantee that the replacement preserves future schemas; V2 needed. |
| F13 | Compiled module `.to(device)` and callable forward execution | [`SwappableDecode.prepare():326–332`](../src/rngd_pd/swap/rngd.py#L326-L332); [`step():410–417`](../src/rngd_pd/swap/rngd.py#L410-L417) | L/S, reachable. Moves the restored wrappers then calls decode and LM graphs with existing resident weight/KV tensors. | Torch-shaped interface on a vendor object, whose implementation is outside this source. P1 provides public generation at a different abstraction; no verified replacement preserving graph argument allocation identity and B2 updates; V6 needed. |
| F14 | Legacy `furiosa.torch.custom_ops.{CompileModule,TacticKernelModule}`, `.dfg.DfgModule`, `.edf.EdfModule`; `furiosa.torch.native_device` setup functions | [`fsdk` legacy module branch:65–71](../src/rngd_pd/engines/fsdk.py#L65-L71) | I when `NEW=False`; unreachable under recorded 2026.4 new branch. `DfgModule` is imported but unused in scoped callers; new branch also imports it at [31](../src/rngd_pd/engines/fsdk.py#L31). | Legacy SDK paths; importable names do not establish current support. No verified public drop-in; migration request V1/V3/V4/V5 as appropriate. |
| F15 | Legacy `module.edf.serialize()`; `furiosa.native_torch.ir.Edf.deserialize`; `CompileModule(EdfModule(edf), weight)` | [`artifact_cache.save():109–116`](../src/rngd_pd/engines/artifact_cache.py#L109-L116), [`load():134–143`](../src/rngd_pd/engines/artifact_cache.py#L134-L143) | C serialization, L deserialization only when `NEW=False`; not in the recorded new-SDK B2 path. | Legacy native artifact and wrapper ABI, unvalidated here. P1 is the documented current Furiosa-LLM artifact direction, not verified compatible with these `.edf/.w` records; V1/V2 needed for a repository migration. |

The 18-file scope also contains dormant callsites: `dsl_batch.Layer.__init__`
constructs six tactic modules ([90–95](../src/rngd_pd/kernels/dsl_batch.py#L90-L95));
`dsl_batch.bench` configures and compiles ([134–139](../src/rngd_pd/kernels/dsl_batch.py#L134-L139));
`kernels.decode_batch.main` configures and compiles ([227–235](../src/rngd_pd/kernels/decode_batch.py#L227-L235));
`BatchDecodeEngine.__init__` and `_build_lmhead` use graph builders/cache and
module `.to()` ([222–231](../src/rngd_pd/engines/decode_batch.py#L222-L231),
[264–291](../src/rngd_pd/engines/decode_batch.py#L264-L291)).
These are D/C paths, not invoked by the binding's `SwappableDecode` factory.
`DfgModule` and `TacticSortingPolicy` have no scoped calls to inventory.

## Public alternatives verified in official Furiosa documentation

Documents were reviewed on 2026-10-31. Versioned 2026.4 URLs are used to align
with the recorded native environment. These are verified capabilities, not
assertions of equivalent behavior in this repository.

| ID | Verified public option and primary source | Relation to this implementation |
|---|---|---|
| P1 | [`fxb build`, cache/add/check, `furiosa-llm serve ... --fxb`](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html); [`furiosa_llm.LLM(..., fxb=...)`, `generate()`, `shutdown()`](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/reference/llm.html) | Public compiled-deployment and generation path. FXB stores kernels separately from HF model assets/weights; compatible changed-weight models can reuse kernels. Not a verified serializer/loader for arbitrary `CompileModule` or a mechanism for transactional partial mutation of this owner's allocations. Requires runtime redesign and new conformance evidence. |
| P2 | [Furiosa-LLM model parallelism](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/model-parallelism.html): TP fixed at FXB compilation; PP/DP selected when serving. | Public PE/model configuration knobs. No verified one-to-one mapping to this repo's `set_fusion(8)` / `MeshKind.Mesh(2)` or stream barrier. |
| P3 | [FXB command reference](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html#command-reference): model build, TP/PP, length and optimization controls. [2026.3 release notes](https://developer.furiosa.ai/v2026.4.0/en/whatsnew/release-2026.3.0.html) introduce TCL / furiosa-kernels. | Public build/kernel direction. These controls do not verify equivalence to `FCConfig`, hint enums, shape cap, `TclModule` grammar conversion or positional wrapper reconstruction. |
| P4 | [Software stack](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html#furiosa-compiler) describes `torch.compile()` with `FuriosaBackend`. | Vendor-documented compiler entry point at overview level. Does not establish import path, exact controls, cache persistence, graph layout or zero-compilation B2 behavior; not a verified drop-in. |

The [2026.4 release migration](https://developer.furiosa.ai/v2026.4.0/en/whatsnew/release-2026.4.0.html)
removes the legacy **Furiosa-LLM model artifact** path in favor of FXB. That
statement does not prove removal of every low-level `furiosa-torch` EDF/IR API
or establish compatibility with the repository's custom `.runnable` archive.

Search coverage included the versioned developer-center index, software stack,
release/component tables, FXB lifecycle, LLM API and model-parallelism docs,
plus official-domain searches for `CompileModule.from_module`, `TclModule`,
`TacticHintConfig`, `_C.config.compiler`, `Runnable.deserialize`,
`module_weight`, `output_reconstructor`, `set_fusion` and `current_stream`.
No exact public replacement contract was verified for F02–F12/F14–F15 in that
coverage. Search misses are a documentation gap, not proof of nonexistence.

## Repository-private engine contract and public Torch operations

These are separate from the Furiosa symbols above:

| Boundary | Exact function/callsite | Classification and implication |
|---|---|---|
| Serving allocation layout | [`RngdBinding._cache_layout():127–152`](../src/rngd_pd/swap/rngd_binding.py#L127-L152), [`_read():288–303`](../src/rngd_pd/swap/rngd_binding.py#L288-L303), [`_write():305–320`](../src/rngd_pd/swap/rngd_binding.py#L305-L320): `engine.shared`, `wlm_d`, `lm_name`, `dev`, `group`, `nl`, `torch` | Repository-owned layout contract, including existing resident tensors, transposed projection rows and LM chunks. These field names do not identify vendor-private APIs. |
| Ownership/lifecycle | [`request():193–205`](../src/rngd_pd/swap/rngd_binding.py#L193-L205), [`apply/rollback():339–410`](../src/rngd_pd/swap/rngd_binding.py#L339-L410), [`_unload():412–430`](../src/rngd_pd/swap/rngd_binding.py#L412-L430): `_lock`, `_sync`, `_generation`, `_failed`, `swap()`, `close()` | Repository-private engine coordination; `_sync` delegates to F03. New runtime alternatives must preserve drain, KV invalidation, undo and quarantine semantics to satisfy the same owner contract. |
| Device readback | [`RngdBinding._read():288–303`](../src/rngd_pd/swap/rngd_binding.py#L288-L303): `.cpu()`, then CPU selection/transpose/clone; `numel`, `element_size`, device/dtype checks | Public Torch tensor surface. Parent allocation is downloaded before CPU row selection. The bounded native evidence exercises the Furiosa backend; public spelling alone does not establish arbitrary-device behavior. |
| Device replacement | [`RngdBinding._write():305–320`](../src/rngd_pd/swap/rngd_binding.py#L305-L320): tensor views, `.t().contiguous().to(device)`, destination `.copy_()` | Public Torch tensor surface. Copies update the existing serving allocation; replacing the tensor object would be a different contract. No direct vendor-internal tensor address API is called. |
| Host bytes/parser | [`_bytes()` / `_digest():73–79`](../src/rngd_pd/swap/rngd_binding.py#L73-L79), [`_file_rows():277–286`](../src/rngd_pd/swap/rngd_binding.py#L277-L286), `delta.py`, `variants.py`, safetensors checkpoint reader and `ModelConfig.inv_freq()` | Host Torch/NumPy, stdlib, safetensors and repository code. Not Furiosa-internal APIs. |

`binding_cli` is a proxy to a persistent owner, not a vendor runtime API.
The optional bare `_tv_patch` import in `dsl_batch:13–14` patches
`torch.library.register_fake` ([`_tv_patch.py:11–30`](../src/rngd_pd/_tv_patch.py#L11-L30));
it is a local Torch workaround, may not resolve under package-only paths, and
does not add a Furiosa-private call. Remaining files in the explicit 18-file
scope contain no additional direct Furiosa imports/calls. Vendor code's own
transitive internals are outside this audit.

## Vendor requests required before claiming a public replacement

| Request | Required answer / acceptance condition |
|---|---|
| V1: artifact persistence | Supported save/load API for the exact `CompileModule` graphs, including native program bytes, compatibility/version guarantees and how to avoid `_inner`; or documented migration from these records to another runtime. Must restore without compiling in B2 load. |
| V2: wrapper state | Supported representation for `module_weight`, `output_reconstructor`, zero-sized outputs and passthrough metadata; stable constructor or public exporter/importer that preserves structured outputs without depending on SDK dataclass layout. |
| V3: compiler configuration | Supported non-private import paths for config and tactic enums on furiosa-torch 2026.4.0, plus equivalent controls for hints and shape limits. Clarify whether stringified enums are stable enough for cache identity. |
| V4: device/synchronization | Supported fusion/mesh and ordinal setup and stream-completion barrier, with lifetime/thread/process scope, ordering and failure semantics for in-place writes, readback and close. |
| V5: custom graph/kernel build | Supported entry point for this graph/DSL input-output signature and symbolic axes, or migration instructions to TCL/FXB/`FuriosaBackend`; identify which semantics and performance require revalidation. |
| V6: resident update integration | Public access/update/readback hooks for the same BF16 serving allocations, whole-request drain, KV invalidation and verified rollback. FXB reuse across changed-weight model loads alone does not answer this requirement. |

No vendor message was sent as part of this audit. Until those answers are
verified, the precise claim is: **public Torch row operations in the adapter,
with inherited SDK-dependent serving and cache internals**. Public deployment
alternatives exist; a public drop-in replacement for this B2 engine has not
been verified.

## Verification

The audit change modifies documentation only. Recompute source/LOC identity
without importing model or SDK modules using
`python experiments/analyze_rngd_binding_loc.py`; compare its JSON to the
checked-in report (raw line-ending hashes are platform-sensitive). Check
relative source links and line ranges against the pinned baseline, and run
`git diff --check`. NPU tests are outside this source/documentation task.

In this clean Windows checkout the two adapter files have CRLF bytes while
their checked-in LOC hashes match the baseline Git blobs with LF bytes.
All physical/nonblank counts match; all other report file hashes match the
working tree. The two differences were checked as line-ending-only. The LOC
report was preserved, rather than silently replacing its source identities.
