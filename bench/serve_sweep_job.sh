#!/usr/bin/env bash
# Run one engine on the pod's one RNGD card and sweep it with bench/vendor_sweep.py.
#
# Used by deploy/k8s/serve-sweep-job.yaml (a Job granted furiosa.ai/rngd: 1) so
# that furiosa-llm and rngd-pd are measured by the same client on the same kind
# of card, one after the other, and the card is released when the Job exits.
#
# Environment:
#   ENGINE     furiosa-llm | rngd-pd
#   OUT        output jsonl (default /out/harness_${ENGINE}.jsonl)
#   CONTEXTS   P grid (default 128,512,1024,2048); CONC concurrency grid (1,2,4,8,16)
#   MAX_TOKENS (64), REPS (2), GREEDY (0: sweep; N: N fixed prompts per P, see vendor_sweep.py)
#   furiosa-llm: FURIOSA_LLM (the furiosa-llm executable), ARTIFACT (artifact dir), LLM_ARGS
#   rngd-pd:     PY (python with furiosa-torch), S, C, RNGD_BUCKETS, B (16), DECODE_GROUP (8),
#                SCHED (continuous), SOLO_ARGS
set -euo pipefail
ENGINE="${ENGINE:?furiosa-llm or rngd-pd}"
OUT="${OUT:-/out/harness_${ENGINE}.jsonl}"
CONTEXTS="${CONTEXTS:-128,512,1024,2048}"
CONC="${CONC:-1,2,4,8,16}"
HERE="$(cd "$(dirname "$0")" && pwd)"
HPY="${HARNESS_PY:-python3}"
hpy() { PYTHONPATH= "$HPY" "$@"; }             # the client needs only numpy + transformers
LOG="${OUT%.jsonl}.server.log"

# The card the device plugin gave this pod, by the name furiosa-smi uses. A
# privileged pod sees every card, so ask the kubelet checkpoint (POD_UID).
NPU=$(PYTHONPATH=/app/src python3 -c 'from rngd_pd.devices import allocated_cards as a; c = a(); print("npu%d" % c[0] if c else "")' 2>/dev/null || true)
if [ -z "$NPU" ]; then
  [ "$(ls /dev/rngd | grep -c 'mgmt$')" = 1 ] || { echo "[job] cannot tell which card is ours"; exit 1; }
  NPU=$(ls /dev/rngd | sed -n 's/^\(npu[0-9]*\)mgmt$/\1/p')
fi
echo "[job] engine=$ENGINE card=${NPU:-?} out=$OUT"
furiosa-smi info || true

PORT=8000
if [ "$ENGINE" = furiosa-llm ]; then
  # its own venv only: the torch in ~/.local is for rngd-pd's baked graphs
  PYTHONPATH= "${FURIOSA_LLM:?}" serve "${ARTIFACT:?}" --served-model-name llama8b \
      --host 127.0.0.1 --port $PORT --devices "npu:${NPU#npu}" ${LLM_ARGS:-} > "$LOG" 2>&1 &
else
  SERVED_MODEL_NAME=llama8b "${PY:?}" -m rngd_pd.serving.cluster --role solo \
      --S "${S:?}" --C "${C:?}" --group 2 --decode-group "${DECODE_GROUP:-8}" \
      --B "${B:-16}" --sched "${SCHED:-continuous}" --chip -1 --port $PORT \
      ${SOLO_ARGS:-} > "$LOG" 2>&1 &
fi
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true' EXIT

for i in $(seq 1 360); do                       # up to 30 min to load
  if ! kill -0 $SRV 2>/dev/null; then echo "[job] server died"; tail -50 "$LOG"; exit 1; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:$PORT/v1/completions \
         -H 'Content-Type: application/json' \
         -d '{"model":"llama8b","prompt":[128000,791,6864],"max_tokens":2,"temperature":0}' || true)
  [ "$code" = 200 ] && break
  sleep 5
done
[ "$code" = 200 ] || { echo "[job] server not ready"; tail -50 "$LOG"; exit 1; }
echo "[job] server ready after $((i * 5)) s"

if [ "${GREEDY:-0}" != 0 ]; then
  hpy "$HERE/vendor_sweep.py" --engine "$ENGINE" --url http://127.0.0.1:$PORT \
      --model llama8b --contexts "$CONTEXTS" --max-tokens "${MAX_TOKENS:-64}" \
      --greedy "$GREEDY" ${GREEDY_TOGETHER:+--greedy-together} --out "$OUT"
else
  hpy "$HERE/vendor_sweep.py" --engine "$ENGINE" --url http://127.0.0.1:$PORT \
      --model llama8b --npu "${NPU:-auto}" --contexts "$CONTEXTS" --concurrency "$CONC" \
      --max-tokens "${MAX_TOKENS:-64}" --reps "${REPS:-2}" --tag "${TAG:-}" --out "$OUT"
fi
tail -5 "$LOG"
