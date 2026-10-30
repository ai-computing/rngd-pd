# Resident NPU update optimization

The user wants the running NPU model retained, with only changed weights or
shape-specific programs updated. Host parsing and artifact reconstruction do
not meet this requirement. The previous provider session and checkout remain
read-only; this branch starts from the independently published da85c80 snapshot.

The user explicitly asked to proceed and to use REA plus the REAP investigation
prompts. Implementation choices and experiments are delegated to the agent.

## Approaches

1. Investigate compiled constant pools and external-weight binding with REA.
   This could preserve fast embedded-weight execution, but exported symbols
   alone do not prove a valid runtime handle or weight/address mapping.
2. Optimize the existing device-input engine. Its unchanged weights already
   stay resident, but its graph performs poorly. Test weight prepacking and
   layout conversion separately before changing its defaults.

Use both investigations to select the smallest evidenced implementation.
Do not use full reload or host preparsing as a substitute for resident updates.

## Evidence and acceptance

- Begin with a bounded toy: resident weights, known outputs, repeated execution,
  B/S changes and a discriminating partial weight update.
- Record preparation, update, first token and steady execution separately.
- Retain unchanged device allocations and avoid recompilation during updates.
- Compare updated outputs with a separately loaded destination, and require
  the changed fixture to differ from the base.
- Validate any selected route on Llama-3.1-8B before claiming model speedup.
- Preserve validation, request locks, failed-update recovery and KV invalidation.
- No global SDK edits; version-sensitive internal APIs must have explicit guards.
- REA records binary hashes and function/address evidence. Separate confirmed
  facts, hypotheses and unresolved ABI/layout assumptions.

## Scope and constraints

The SDK environments are rebel 0.11.0 / torch-rbln 0.2.2 and
furiosa-torch 2026.2.0. No promise of arbitrary binary-byte patching is made.
Only allocated experiment devices may execute probes; use hub.orca.run_remote.
Preserve evidence and clean up only resources created by this experiment.
Python 3.10+, no new production dependencies.
