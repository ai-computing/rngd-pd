# Standalone prepared serving and native prewarm

The hardware SDK environment has no vLLM. The user-authorized integration must
serve real whole-request KV without test stubs or changes to the shared SDK.
The existing vLLM plugin remains a genuine subclass when vLLM is installed.

Extract its shared transfer implementation into a dependency-free worker
connector with its own role/metadata types. The vLLM adapter retains its public
constructor and metadata/base class contract and reuses those same methods.
Cluster workers use only the standalone connector. Missing vLLM produces a
clear error only for the vLLM adapter; no fake vLLM modules are supplied.
Installing vLLM would change the shared hardware environment; duplicating the
transfer logic would create divergent wire behavior. Both are outside scope.

Optional prepared config `warmup` is an exact boolean, default false. When true,
startup selects every configured variant, allocates private blank KV and runs
one actual engine step at safe token/position zero before readiness. Private
KV is discarded after each attempt, including failure or cancellation. Any
startup failure closes the initialized engine and propagates; readiness is
never published. Existing atom-inputs materialization remains unchanged.

No new dependencies, remote processes or device allocations are authorized by
this code task. Verify fresh producer/consumer processes with all vLLM imports
blocked and genuine shared-memory transfers. Use CPU adapters with only SDK
boundaries substituted for warmup tests. Hardware owner will replay real native
prompt K/V and compare HTTP output against fresh tokens; synthetic KV is not
prefill evidence. The parent supplied this design as an explicit implementation
instruction, so no additional design approval menu is needed.
