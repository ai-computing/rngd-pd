# Onggi preparation to native prepared HTTP serving

2026-10-30 · RNGD integration correctness, not a performance comparison.

The real init-container → standalone prepared worker → HTTP decode path passed
on one allocated RNGD, using SDK 2026.4. The worker warmed before readiness and
handled three batches of four requests with resident projection/LM buffers and
compiled graphs retained. This validates one B4/S256/group2 configuration.

The [complete method and bilingual report](https://github.com/ai-computing/onggi/blob/8f0b4bc103cc1608a7288bd14c59f67a292d3d74/docs/native-serving-2026-10-30.md)
and [sanitized evidence for both attempts](https://github.com/ai-computing/onggi/blob/8f0b4bc103cc1608a7288bd14c59f67a292d3d74/docs/data/native-serving-20261030.json)
are pinned to an immutable Onggi documentation commit.

## Frozen conditions and result

| Item | Observed value |
| --- | --- |
| Runtime source executed | `fafd820fead20ed51dcb7f60ebb5e7dc7d90b2f1` |
| Onggi source executed | `466ef918db08a8722326b6f36f5f39597e8a63de` |
| Model | Llama-3.1-8B, B4/S256/group2, fusion8/mesh2 |
| SDK / Torch | 2026.4.0 / 2.10.0+cpu |
| Resources | One plugin-assigned RNGD; CPU request4/limit12; RAM request64Gi/limit128Gi; deadline7200s |
| Prepared config | `host_workers:2`, `warmup:true`, intra-op4, shm BF16 wire |
| File delivery | Two precompiled graphs, total984729B, plus config/provenance |
| External checkpoint | Six read-only files, total16060581181B; not transferred by Onggi |
| HTTP result | 12 requests /96 tokens equal fresh references |
| Token accounting | 12 supplied first tokens;84 native-produced suffix tokens |
| Worker native calls | 17 before readiness →136 →255 →374 |
| Upload / graph load / compile | 1 /1 /0 throughout |
| Retained identities | All112 shared projection buffers, one LM buffer, compiled modules |
| Admission checks | Checkpoint mismatch409; context overflow400 |
| Cleanup | Both final containers exit0; worker child stopped after checks; all3 owned Pod UIDs and namespace independently absent |

The native producer replays the original four eight-token prompts and exports
actual full S256 K/V arrays. All32 fresh reference tokens also match the earlier
independent public fresh-base record. Bitwise BF16 wire roundtrips cover prefix
and tail. The producer exits before the fresh HTTP worker starts; no vLLM test
modules or native/transport/token substitutes are used. Onggi's real init selects
the matching manifest, rejects the wrong-firmware candidate and verifies all
delivered files before the main container consumes them read-only.

## Attempts and measurement limits

Attempt1 incorrectly sliced KV to eight prompt columns in the validation harness.
Its four requests returned400 before native continuation. This invalid serving
attempt is preserved separately. Attempt2 changed only the harness to export
the full actual S256 arrays with roundtrip checks, keeping frozen code, profile,
checkpoint and resource conditions. The misleading source comment was corrected
after the run; all runtime `src/` Python ASTs remain identical to frozen source.

The successful run measured combined native construction/graph load/warmup at
13.988s and worker-process start to HTTP readiness at16.424s. **Warmup alone was
not timed.** Retry file preparation took0.022s on an already used local store/cache.
These are observer-instrumented diagnostics; no cold-load or speedup inference
is valid. The earlier [host staging comparison](2026-10-30-rngd-host-staging.md)
has separate conditions and samples.

Successful negative-request assertions checked native-counter and KV-file-list
preservation; individual before/after negative snapshots were not saved. Native
checkpoint, graph and512 KV payload files remain remote-only; local verification
rehashes the205 exported small evidence files and checks canonical source hashes,
tokens, profiles, manifests, identities and cleanup records. The private snapshot
manifest SHA256 is
`14a0eebb5f0f8bc850b0705128f43b48ee39564a8ffd7e73e47266356a43f94c`.

This is native HTTP decode from real prompt/KV replay. Live prefill/gateway
handoff, multiple native B/S choices, ATOM HTTP startup, installed-vLLM runtime,
UCX recovery, production rollout and performance/SLO evaluation remain separate
gates. `atom-constants` remains a Python API/validation-harness capability, not a
prepared HTTP factory. No shared SDK packages or foreign workloads were changed.
