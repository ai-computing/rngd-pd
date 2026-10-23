# Safe serving and prepared delta integration

The user accepted the review's three-stage direction: repair current serving and
transport correctness, connect existing resident swap engines, then enable
prepared request-boundary variant selection. Work only in this independent
Codex checkout; the prior provider session remains read-only.

## Approach

Keep `serving.cluster` as the deployed entry point. Reuse swap engines and their
artifact cache, rather than adding another server or a new dependency. onggi
continues to provide artifact delivery and delta planning. A resident worker
owns model weights, selected graphs and request KV. Native-v4's ownership and
bounded-admission rules inform the existing scheduler; replacing the entire
server with native-v4 would unnecessarily change model/runtime interfaces.

## Requirements

- Fix the reviewed batching, admission, request-ID, prompt validation, transport,
  UCX allocation, connector metadata, bundle race and KV ownership defects.
- Reject overload before allocating/loading full KV. Bound ahead-loaded KV and
  support cleanup when queued/streaming requests are abandoned.
- Shape changes happen at drained request boundaries. Use prepared graphs only;
  no compile-on-request or migration of in-flight KV across models/shapes.
- RNGD shares resident weights across prepared B/S variants. ATOM embedded-weight
  bundles remain the primary shape path; device-input delta loading stays opt-in
  because its decode cost is greater on the tested SDK.
- An uncoordinated decode-only checkpoint update must not accept old prefill KV.
  Checkpoint changes therefore require explicit draining and matching producer
  identity; do not silently make an unsafe HTTP update endpoint.
- Use existing Python >=3.10, NumPy and vendor stacks; no new dependencies.
- Preserve the chart's standalone relay copy exactly when its source changes.
- Add regression tests before fixes, verify CPU suites, and document hardware
  checks not performed. No deployment, merge, or previous-session modifications.

## Acceptance

Mixed output lengths cannot advance finished lanes beyond context; one-token
requests return exactly one token; replicas produce distinct KV request IDs;
overload is finite; malicious/truncated/stalled transport payloads fail safely;
bundle release races and foreign engine KV are rejected. Prepared serving
selects the smallest compatible ready shape and never changes active requests.

Hardware token parity, actual SDK loading latency, and DMA cancellation safety
need separate real-device verification before broad deployment.
