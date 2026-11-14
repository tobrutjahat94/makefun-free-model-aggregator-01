#!/usr/bin/env bash
# Run one practice trial locally against the sealed env + scorer, exactly as the
# server does (process driver). Usage: ./run-trial.sh <trial> [port]
#   trial: 0, 1 or 2 (practice/trial-<n>)
set -u
KIT="$HOME/workspace/makefun/kits/free-model-aggregator-01/free-model-aggregator-01-kit"
ENTRY="$HOME/workspace/makefun/entry-free-model-aggregator-01"
TRIAL="${1:-0}"
PORT="${2:-18080}"
STATE="/tmp/fma-test/trial-$TRIAL"
rm -rf "$STATE"; mkdir -p "$STATE"

echo "== env: trial-$TRIAL on port $PORT"
MAKEFUN_ENV_DATA_DIR="$KIT/practice/trial-$TRIAL/input/env" \
MAKEFUN_STATE_DIR="$STATE" \
MAKEFUN_ENV_PORT="$PORT" \
node "$KIT/program/env/server.mjs" >"$STATE/env.log" 2>&1 &
ENV_PID=$!
for i in $(seq 1 50); do [ -f "$STATE/ready" ] && break; sleep 0.2; done
cat "$STATE/ready" 2>/dev/null; echo
[ -f "$STATE/ready" ] || { echo "env did not become ready"; cat "$STATE/env.log"; kill $ENV_PID 2>/dev/null; exit 1; }

echo "== entry: main.ts against the env"
MAKEFUN_ENV_URL="http://127.0.0.1:$PORT" \
MAKEFUN_PROMPT_FILE="$KIT/practice/trial-$TRIAL/input/prompt.txt" \
MAKEFUN_TRIAL_SEED="test" \
timeout 120 node "$ENTRY/main.ts" >"$STATE/entry.log" 2>&1
ENTRY_RC=$?
echo "entry exit=$ENTRY_RC"
cat "$STATE/entry.log"

echo "== env: SIGTERM for the state export"
kill -TERM $ENV_PID 2>/dev/null
wait $ENV_PID 2>/dev/null
ls -la "$STATE/export/" 2>/dev/null
[ -f "$STATE/export/state.json" ] || { echo "NO state.json"; tail -20 "$STATE/env.log"; exit 1; }

echo "== scorer"
MAKEFUN_OUTPUT_DIR="$STATE/export" \
MAKEFUN_REFERENCE_DIR="$KIT/practice/trial-$TRIAL/reference" \
node "$KIT/program/scorer/score.mjs"
echo "== per-request detail"
python3 - "$STATE/export/state.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
for r in s['requests']:
    failed = [c['name'] for c in r['checks'] if not c['pass']]
    print(f"{r['id']} valid={r['valid']} model={r['model']} rating={r['rating_milli']} calls={r['upstream_calls']} fails={[ (f['provider'], f['status']) for f in r['failures'] ]} failed_checks={failed} err={r['error']}")
print('totals:', s['totals'])
print('catalog_route:', s['catalog_route'])
EOF
