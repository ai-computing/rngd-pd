# RNGD resident delta optimization

The user requested further Furiosa RNGD improvements after the ATOM constant
refresh work. Improve resident checkpoint updates, especially small changes,
without reloading unchanged weights or recompiling decode graphs. Preserve
fresh-checkpoint numerics and transactional failure behavior.

## Existing evidence

`SwappableDecode` already retains device weights and compiled B/S variants.
The previous 8B B4/S256 test uploaded one 117,440,512-byte projection tensor,
with 128 generated tokens exactly equal to fresh references. The planner
computes changed checkpoint rows, but the updater ignores them. It transposes
the entire new tensor and old rollback tensor, uploads a temporary device
tensor, then copies it into the resident layer.

Checkpoint rows become columns of transposed projections. Existing native KV
code warns that naive strided tensor slices can bridge through the host.
Consequently, CPU slice tests do not establish efficient native row updates.
LM-head rows are contiguous within its padded/chunked resident layout.

## Approaches and decision

1. Use documented CPU-to-existing-device copies to avoid temporary device
   allocations. Small and broadly applicable; improvement must be measured.
2. Honor refined row ranges, using native column transfers for projections and
   contiguous row transfers for LM head. This reduces host preparation and
   logical transfer payload, but requires a hardware-proven native copy route.
3. Recompile graphs around a new weight layout or native scatter kernel.
   Larger implementation and cache changes; reserve for demonstrated need if
   native copy support is insufficient.

Agent decision: probe approaches 1/2 first and use the smallest route with actual
native evidence. Do not ship a strided fallback that merely hides full copies.
Use the SDK HAL sequencer only if its layout/alias behavior and completion are
demonstrated on allocated hardware. Fix cancellation/rollback hazards in the
touched updater regardless of which transfer route wins. Keep normalization
and host embedding handling conservative; no arbitrary byte-offset claim.

## Constraints and correctness

- Work in the new isolated clone based on `da85c80`; prior provider session and
  prior checkout are read-only. ATOM optimization remains in its own branch.
- Remote operations only through `hub.orca.run_remote`; compute on rebellion,
  Kubernetes control on dell_0. Use owned allocated devices, immutable input
  fixtures, explicit Pod UIDs, bounded jobs and ownership-checked cleanup.
- Retain the validated furiosa-torch 2026.2.0 / Torch 2.11 CPU/cache pairing for
  comparisons. Never patch installed SDKs or change global agent configuration.
- No new production dependency. Existing checkpoint and `load_delta` callers
  remain compatible; missing row metadata retains whole-tensor behavior.
- Validate real checkpoint layouts and supplied row ranges before mutation.
  Preserve FP32 q scaling before BF16 conversion and LM padding/chunk layout.
- Stage new and rollback data before first device mutation. On any exception or
  interruption restore prior state or disable serving. Old KV must not remain
  usable across a potentially changed device state.
- Report logical payload separately from measured bus traffic, and separate
  live onggi/planning, host preparation, native copy and first response.

## Validation and completion

CPU tests use real tensors/checkpoint files and only fake the native boundary.
Hardware must demonstrate updates through the original resident allocation,
unchanged neighbors, native execution and fresh-reference output parity.
Compare frozen baseline and optimized implementations under matching resources;
include full-tensor and discriminating sparse changes where the chosen route
supports them. Record every sample, failed controls, SDK/source hashes and scope.
Complete independent review, appropriate CI, documentation, a reviewable PR,
and removal of owned active resources. No production rollout is implied.
