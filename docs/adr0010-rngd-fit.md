# ADR-0010 RNGD compatibility and remaining work

The RNGD performance path consumes the exact `onggi.switchplan.v1` produced by
[onggi PR76 at 2d84ac3](https://github.com/ai-computing/onggi/tree/2d84ac39374d71086436e65b967a4f718876a2f3).
It does **not** yet implement or pass that proposal's B2 device-binding contract.
E6/E8 native switch measurements and E7 row-update measurements establish their
recorded performance and token checks; they do not establish B2 conformance.

`parse_onggi_switchplan()` preserves manifest identities and source metadata,
validates complete ordered target partitions and classification totals, rejects
unknown schemas and unsafe/duplicate paths, and accepts additive fields. The
existing range loader uses canonical `elsewhere`/`new` spans, without requiring
the legacy `changed` field. Positive `same` bytes require a paired base file,
enforcing the documented meaning even though the JSON Schema omits that
semantic implication. Unversioned legacy input remains supported.

The parser adds 95 lines and changes three existing lines in `delta.py`; its
139 new test cases and existing tests passed in the full local suite:
590 passed, 8 skipped, 4 subtests. Exact-source `f0f08bf` CI41051893033 was read:
596 passed, 2 skipped, 4 subtests. These are CPU parser/harness checks. All four
unmodified upstream golden switch plans were separately accepted. E7 uses
actual PR76 CLI output from a separately built binary, not hand-converted JSON.

## Requirements still unmet

| ADR-0010 property | Existing serving path | Required implementation |
|---|---|---|
| Five JSON operations and resident manifest identity | Python engine methods; no five-operation owner protocol | Keep one serving process as allocation owner; expose a thin JSON proxy and explicit manifest/state transitions. |
| Pure plan covers every non-`same` artifact byte | `plan_update(refine=True)` reads checkpoints, drops equal rows and only maps safetensors data | Separate contract planner using resident mapping metadata. Cover rounded device units, including chunk false positives, or refuse. No target reads during contract planning. |
| Actual device readback SHA256 in original artifact byte space | `apply_delta()` synchronizes writes and checks later token output, with no per-unit readback digest | Read actual device values, reverse only documented lossless layouts, serialize original artifact bytes and compare digests before publishing target identity. |
| Rollback after the last successful apply | Exception recovery uses the old checkpoint; restore values are local to `apply_delta()` | Retain a journal of actual old resident values plus manifest/state. Verify restoration and unload on failed restoration. |
| Drain every affected request | Whole `generate()` calls use a lock, but standalone `step()` can span an unfinished request | Serving admission gate and active-request accounting covering staging, pending work, apply, verification and restoration. |
| Rewrite only planned units | Density/fragmentation fallback and group replacement can enlarge writes | Publish final expanded units before execution, or refuse. |
| Runtime-valid native conformance fixtures | Upstream kit supplies arbitrary `model.bin` bytes and a tiny config | Allow caller-provided valid runtime artifacts with the same coverage, manifest, verification and rollback assertions. Generic byte storage is a different binding scope. |

Code inspected: [planner](../src/rngd_pd/swap/delta.py),
[weight transforms and update](../src/rngd_pd/swap/rngd.py), and the exact
[contract](https://github.com/ai-computing/onggi/blob/2d84ac39374d71086436e65b967a4f718876a2f3/docs/adr/0010-device-binding-contract.md).
This is a source compatibility assessment, not a failed native B2 test run.

## Artifact verification has a real representational limit

The engine scales query weights then converts to BF16, converts other source
dtypes to BF16, and scales/tiles norms. These conversions can lose information;
device values alone cannot in general reproduce every original artifact byte.
Embeddings and original norms also have host-side representations. Rehashing
the target file or a host shadow is not device readback. Restarting does not
make a lossy representation invertible.

A concrete initial serving subset is unchanged metadata plus BF16 weights whose
resident representation is only transposed/grouped or padded: K/V/O and MLP
projections, and BF16 LM rows. A future binding can invert those layouts, strip
padding and hash actual readback in artifact order. It must refuse updates to
scaled query weights, lossy dtype conversions, headers/config or other unmapped
bytes. Reload verification semantics for such unsupported artifacts need an
explicit contract decision. No CPU fake or unused mirrored device buffer can
establish conformance of this model-serving path.

The upstream kit also needs to match verification units against the planned
coverage: its current `verify_units()` iterates the supplied list and accepts
an empty list. [onggi PR80](https://github.com/ai-computing/onggi/pull/80) fixes missing, duplicate
and unexpected verification intervals, including empty-file reload and independent
device byte accounting. Twelve stdlib regressions and the reference fake B1/B2
cases pass locally. Runtime-valid fixture hooks remain in
[onggi#79](https://github.com/ai-computing/onggi/issues/79). Corruption, drain and
restoration-failure cases still need runtime tests even when a synthetic binding passes.

## Public API audit

The existing SDK2026.4 serving/cache path has internal dependencies that must
be flagged before any public-API-only claim:

- [fsdk.py](../src/rngd_pd/engines/fsdk.py) serializes through
  `module.runnable._inner.serialize()` and falls back to
  `furiosa.torch._C.config.compiler` imports.
- Cache reload reconstructs `CompileModule` from `ir.Runnable.deserialize()`;
  its documented public stability has not been established here. The older
  cache loader also imports `furiosa.native_torch.ir`.

No vendor format was decoded for these experiments. Public Torch row-copy and
readback methods in a future binding would not by themselves remove the cache
path's internal dependencies. Preserve this distinction in paper claims.

Next steps are a runtime-valid fixture hook and stricter verification coverage
in the neutral kit, then an in-process owner for the lossless subset with real
readback, retained rollback and native corruption/drain tests. These require
new implementation and native evidence; B2 remains unclaimed until they pass.
