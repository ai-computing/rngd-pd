# RNGD initial weight loading investigation

The user authorized continued work on the remaining loading latency. Prior
SDK2026.4/Torch2.10 group2 8B constructor runs take roughly19–21seconds;
upload_weights_s includes CPU group transforms/stack/padding, native .to calls
and stream completion. It is not a measurement of DMA alone.

Separate these costs on an owned allocated card before choosing a change.
Then implement the smallest improvement supported by measured evidence.
Keep q scaling in float32 before BF16 rounding, dense weight layouts, compiled
graph/cache keys, native output parity, row updates and interruption rollback.
No speculative SDK/compiler patches or live FXB replacement. Shared SDK,
driver, source fixtures, previous evidence and provider checkout stay unchanged.

Alternatives: reduce CPU layout copies, reduce native upload calls using a
supported SDK path, or retain the current implementation if neither improves
the actual end-to-end load. The executor chooses after profiling; no promise
of a speedup. Local CPU strided-stack and per-slice copy prototypes were slower
and are rejected, not integrated. Any host-memory or compatibility tradeoff
must be measured and recorded. Full grouping default remains unchanged.

Acceptance: exact CPU weight equivalence and meaningful resource/behavior
regressions before implementation; full CPU suite and one fresh branch review;
same-stack fresh-process timing, native8B token parity and retained update/
rollback behavior. Keep all samples and source/artifact/fixture hashes.
Hardware owner exclusively allocates/cleans one card, bounded deadline and UID
checks. Publish code/evidence/docs on a branch/PR and verify final CI.

Measured selection: original 18.746s constructor includes15.381s host builders
versus1.946s in113device transfers. Same-host CPU prototypes choose two pure
tensor transform workers: q205->114ms, gate220->143ms, down220->146ms, including
pool setup/join. All36 samples and12checks preserve exact BF16 bits.

Implement host_workers=1(default) or2(opt-in), also in RNGD prepared config.
Fetch checkpoint tensors on the caller thread, transform only on workers with
thread-local no_grad, retain order and stack normally. Use an executor only
during initial upload and join it on every exit. Leave delta/group rebuilds
serial and do not change q arithmetic or global Torch thread settings. At
intra-op4, two workers can use8CPU threads; operators choose the CPU budget.
