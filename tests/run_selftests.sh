#!/usr/bin/env bash
# NPU-free regression suite for the RNGD × llm-d package (Phases 1-5).
# Portable: resolves the package root from this script's location, runs every module self-test that
# does not require the NPU, reports a single verdict. NPU-dependent checks (accuracy 9/9, ms/token,
# artifact reload) run separately on hardware.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # package root (parent of tests/)
cd "$ROOT"
PASS=0; FAIL=0
run() {
  local name="$1"; shift
  if out=$("$@" 2>&1); then
    if echo "$out" | grep -q "PASS"; then echo "  ✅ $name"; PASS=$((PASS+1));
    else echo "  ❌ $name (no PASS marker)"; echo "$out" | tail -3; FAIL=$((FAIL+1)); fi
  else echo "  ❌ $name (exit $?)"; echo "$out" | tail -3; FAIL=$((FAIL+1)); fi
}
echo "=== RNGD × llm-d — NPU-free regression suite ==="
echo "    root: $ROOT"
echo "-- Phase 1: connector + transport --"
run "kv_transport"          python3 connector/kv_transport.py
run "rngd_kv_connector"     python3 connector/rngd_kv_connector.py
run "connector_tcp(xproc)"  python3 tests/test_connector_tcp.py
echo "-- Phase 2: gateway --"
run "registry"              python3 gateway/registry.py
run "kv_events"             python3 gateway/kv_events.py
run "epp"                   python3 gateway/epp.py
run "gateway_http"          python3 gateway/server.py
echo "-- Phase 4: dim-parameterized DSL --"
run "dsl_gen"               python3 runtime/dsl_gen.py
echo "-- Phase 5: serving + artifact cache --"
run "bucket"                python3 serving/bucket.py
run "batching"              python3 serving/batching.py
run "serving_app(mock)"     python3 serving/serving_app.py --role all --mock
run "artifact_cache_key"    python3 serving/artifact_cache.py
echo "=========================================="
echo "  PASS=$PASS  FAIL=$FAIL"
[ "$FAIL" -eq 0 ] && echo "  ALL GREEN ✅" || echo "  SOME RED ❌"
exit "$FAIL"
