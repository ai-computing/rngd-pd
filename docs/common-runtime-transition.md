# One caller for RNGD and ATOM transitions

No driver patch or new vendor SDK API is required for this operation. Both
existing owners implement the same explicit caller:

```python
# First authenticate immutable target files against the artifact manifest and
# generate/verify a complete, truthful onggi.switchplan.v1 (including unchanged ranges).
result = owner.transition(verified_switch_plan, verified_target_directory)
assert result["resident"]["manifest"] == verified_switch_plan["manifest"]
with owner.request() as engine:
    output = engine.generate(prompts, n_new)
```

The persistent-owner CLI exposes the same operation:

```sh
python -m rngd_pd.swap.binding_cli --socket /run/binding.sock \
  transition --target /artifacts/target < verified-switchplan.json
```

This is a caller extension, not a replacement for the five ADR-0010 operations
or a claim of automatic CSI-to-runtime integration. Parsing a plan does not
authenticate its artifact identities or prove its classification is truthful.
The endpoint is a trusted local-worker interface, not a multi-tenant service.

| Stage | Operation and invariant |
|---|---|
| Validate | Strictly parse switchplan; require its base to be ready and resident |
| Discover | describe(target), then plan(changes) under the same owner lock |
| Decide | RNGD: admitted BF16 units or explicit reload; ATOM B1: explicit reload |
| Execute | units calls checked apply; reload calls existing load lifecycle |
| Publish | Require ready target identity and, for units, successful target verification |

The owner lock excludes inference, concurrent transitions, load and close.
Serving must use request() for a whole request; retaining a bare engine reference
and calling it outside request() bypasses that guarantee. A stale base is rejected
even if a standalone B1 plan would select reload. Initial admission uses load().
An admitted update failure propagates; the caller does not silently retry/reload.
RNGD's apply retains its existing device-readback undo and recovery semantics.
Successful B2 transitions keep rollback available. B1 apply/rollback still raise
NotImplementedError. Failure during reload can leave empty residency, as before.

## ATOM host parsing reuse

Before replacing a default public AtomSession, the owner streams the new graph
SHA-256. If it matches the previously admitted public graph, it retains that
session's parsed RBLNCompiledModel across old-runtime teardown and passes it to
the existing AtomSession(compiled=...) path. Only one device runtime is live.
Changed graph bytes reparse; explicit close drops the parsed object too.
Injected factories and experimental constant-engine attachment do not use this
reuse path. The fresh graph hash and normal metadata validation remain.

graph_evidence().compiled_reused reports this host-object reuse. It is not device
readback, retained device weights, a selective constant update, or a device-transfer
counter. Runtime creation still occurs. Prior native B1 timing used the earlier
loader and is not a timing result for this optimization. New native timing and
same-target output parity must be measured before claiming a speedup.

## Where vendor cooperation is still needed

RNGD's existing bounded BF16 B2 uses real serving allocations, readback and
retained undo. It interprets supported safetensors rows; arbitrary compiled FXB/EDF
bytes cannot be treated as device addresses. The inherited graph/cache paths
still have private Furiosa SDK dependencies; this caller does not remove them.

ATOM's experimental ConstantWeightDecode can refresh constants using internal
transform groups and retained CPU parameter identities. The inspected installed
SDK source/stubs expose apply_extern_transforms, apply_host_params and dependency
groups. The inspected PyRblnSyncRuntime declaration has KV-cache fetch/update but
does not declare general constant-buffer readback. That is a bounded static
inspection, not proof that no possible native/driver path exists. Host copies or
an applied-group count alone do not satisfy B2 actual-device verification.

Consequently this change implements common orchestration and public host-parse
reuse on ATOM, not ATOM B2. SDK files, native binaries and drivers are unchanged.
A separately negotiated stable update/readback/recovery API would make the
experimental path supportable without private ABI assumptions.

Official public reference (checked 2026-11-02; documentation version is not an
upgrade of the installed SDK):
https://docs.rbln.ai/v0.12.0/software/api/python/python_api.html
https://docs.rbln.ai/v0.12.0/software/api/deep_dive.html

CPU tests exercise real owner classes with injected vendor boundaries, actual
BF16 CPU buffers, transport subprocesses, failed writes/restoration, wrong base,
request exclusion, graph identity changes and parsed-object lifetime. These are
not new native measurements. Prior bounded native reports remain unchanged.
