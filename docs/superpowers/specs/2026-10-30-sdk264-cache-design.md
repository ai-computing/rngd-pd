# SDK 2026.4 resident cache migration

User request: implement the 2026.4 migration and report Onggi's current scope.
Existing RNGD row-update code is frozen at 719c09b (published with 7525204).

The 2026.4 SDK replaces the old EDF container with native `ir.Runnable`.
`CompileModule` additionally owns output reconstruction metadata. Persisting
only executable bytes and weights can therefore silently change output shape.

Use native serialize/deserialize methods without interpreting their bytes.
One atomic `.runnable` ZIP holds metadata, the native blob and a trusted Torch
state pickle containing weights plus the output reconstructor. Validate format,
SDK and Torch versions before deserializing either payload. New caches include
Torch in their directory; the old 2026.2 `.edf`/`.w` path remains unchanged.
The SDK adapter confines access to the Python wrapper's private `_inner` field;
this is the installed 2026.4 serialization boundary, not a stable vendor promise.

Alternatives: recompile on every restart fails prepared-serving requirements;
renaming old EDFs is incompatible; implementing the complete Furiosa-LLM FXB
runtime exceeds this custom engine's needs. The native Runnable cache is the
smallest viable custom-kernel path. No claim of live Furiosa-LLM FXB replacement.

Update bake/cache-probe callers so they no longer assume two EDF files. Preserve
existing row selection, q scaling and transaction behavior. Hardware validation
must prove cache reload in a fresh process, nontrivial outputs, resident 8B fresh
token parity and retained allocations/compiled modules, with synchronized timing.
Use only owned allocated hardware; leave shared SDK/driver/provider files alone.
