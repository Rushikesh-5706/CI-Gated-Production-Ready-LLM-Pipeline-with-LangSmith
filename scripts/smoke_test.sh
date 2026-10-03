#!/usr/bin/env bash
set -euo pipefail

BASE="http://localhost:${PORT:-8000}"
PASS=0
FAIL=0

check() {
    local label="$1"
    local ok="$2"
    if [ "$ok" = "true" ]; then
        echo "PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $label"
        FAIL=$((FAIL + 1))
    fi
}

echo "waiting for $BASE/health ..."
for i in $(seq 1 30); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

HEALTH=$(curl -sf "$BASE/health")
check "/health returns 200" "$([ -n "$HEALTH" ] && echo true || echo false)"

PROMPTS=$(curl -sf "$BASE/prompts")
check "/prompts returns 200" "$([ -n "$PROMPTS" ] && echo true || echo false)"

# summarize_text v1
RESP=$(curl -sf -X POST "$BASE/invoke/summarize_text" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "The Amazon rainforest covers over 5.5 million square kilometers across nine countries in South America. It contains roughly 10 percent of all species on Earth and produces about 20 percent of the world oxygen.", "prompt_version": "v1"}')
OUTPUT=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('output',''))" 2>/dev/null)
TRACE=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('trace_url',''))" 2>/dev/null)
check "summarize_text v1 has output" "$([ -n "$OUTPUT" ] && echo true || echo false)"
check "summarize_text v1 trace_url starts with https" "$(echo "$TRACE" | grep -q '^https://' && echo true || echo false)"

# summarize_text v2
RESP=$(curl -sf -X POST "$BASE/invoke/summarize_text" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "The Amazon rainforest covers over 5.5 million square kilometers across nine countries in South America. It contains roughly 10 percent of all species on Earth and produces about 20 percent of the world oxygen.", "prompt_version": "v2"}')
OUTPUT=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('output',''))" 2>/dev/null)
check "summarize_text v2 has output" "$([ -n "$OUTPUT" ] && echo true || echo false)"

# extract_entities v1
RESP=$(curl -sf -X POST "$BASE/invoke/extract_entities" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "On 14 March 2023, Dr. Yuki Tanaka presented findings at the Geneva Climate Forum.", "prompt_version": "v1"}')
check "extract_entities v1 returns 200" "$([ -n "$RESP" ] && echo true || echo false)"

# extract_entities v2 - check JSON structure
RESP=$(curl -sf -X POST "$BASE/invoke/extract_entities" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "On 14 March 2023, Dr. Yuki Tanaka presented findings at the Geneva Climate Forum.", "prompt_version": "v2"}')
HAS_KEYS=$(echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
o = d.get('output', {})
if isinstance(o, dict) and 'names' in o and 'dates' in o and 'locations' in o:
    print('true')
else:
    print('false')
" 2>/dev/null)
check "extract_entities v2 has names/dates/locations" "$HAS_KEYS"

# unknown version -> 404
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/invoke/summarize_text" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "test", "prompt_version": "v99"}')
check "unknown version returns 404" "$([ "$STATUS" = "404" ] && echo true || echo false)"

# unknown prompt -> 404
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/invoke/nonexistent" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "test", "prompt_version": "v1"}')
check "unknown prompt returns 404" "$([ "$STATUS" = "404" ] && echo true || echo false)"

echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
