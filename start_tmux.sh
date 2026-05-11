#!/usr/bin/env bash
# Launch a multi-process MLX serving setup inside a tmux session, one pane per
# backend plus a routing proxy on top. Same model layout as start_multi.sh —
# this just makes each backend's stdout visible in its own pane.
#
# Usage:
#   ./start_tmux.sh                                  # default mini+small split
#   ./start_tmux.sh --detach                         # don't auto-attach
#   ./start_tmux.sh --kill                           # tear down session
#
# Env overrides (same as start_multi.sh):
#   MODELS="mini:8810,small:8811,medium:8812" ./start_tmux.sh
#   PROXY_PORT=9000 ./start_tmux.sh

set -euo pipefail

SESSION="${SESSION:-mlx-multi}"
MODELS="${MODELS:-mini:8810,small:8811}"
PROXY_PORT="${PROXY_PORT:-8800}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# --- handle --kill ---
if [ "${1:-}" = "--kill" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "killed $SESSION" || echo "no session $SESSION"
    # Cleanup any stragglers on tracked ports
    IFS=',' read -ra PAIRS <<< "$MODELS"
    for pair in "${PAIRS[@]}" "x:$PROXY_PORT"; do
        port="${pair##*:}"
        pid=$(lsof -ti :"$port" 2>/dev/null || true)
        [ -n "$pid" ] && kill -9 $pid 2>/dev/null && echo "killed pid on :$port"
    done
    exit 0
fi

# --- pre-flight ---
command -v tmux >/dev/null || { echo "tmux not installed. brew install tmux"; exit 1; }
[ -d "$VENV_DIR" ] || { echo "missing $VENV_DIR — run setup.sh first"; exit 1; }

# --- if session exists, attach ---
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session $SESSION already exists — attaching"
    [ "${1:-}" = "--detach" ] || tmux attach -t "$SESSION"
    exit 0
fi

# --- check ports ---
for pair in $(echo "$MODELS" | tr ',' ' ') "proxy:$PROXY_PORT"; do
    port="${pair##*:}"
    if lsof -nP -iTCP:"$port" 2>/dev/null | grep -q LISTEN; then
        echo "port $port already in use — use ./start_tmux.sh --kill or free it manually"
        exit 1
    fi
done

# --- create session ---
echo "creating tmux session '$SESSION'"

# Use pane IDs (%0, %1, ...) — independent of pane-base-index tmux config.
IFS=',' read -ra PAIRS <<< "$MODELS"
n=${#PAIRS[@]}

# Create initial session with first pane, capture its id
declare -a PANE_IDS=()
PANE_IDS+=("$(tmux new-session -d -s "$SESSION" -n stack -x 220 -y 60 -P -F '#{pane_id}')")

# Split into (n) total model panes
for ((i=1; i<n; i++)); do
    PANE_IDS+=("$(tmux split-window -v -t "${PANE_IDS[i-1]}" -P -F '#{pane_id}')")
done

# One more pane for the proxy
PROXY_PANE="$(tmux split-window -v -t "${PANE_IDS[n-1]}" -P -F '#{pane_id}')"
tmux select-layout -t "$SESSION:stack" even-vertical

# Send commands to each model pane
for ((i=0; i<n; i++)); do
    pair="${PAIRS[$i]}"
    tier="${pair%%:*}"
    port="${pair##*:}"
    tmux send-keys -t "${PANE_IDS[i]}" \
      "cd $SCRIPT_DIR && source .venv/bin/activate && \
       MLX_PORT=$port python3 serve.py --preload $tier" C-m
done

# Proxy pane: wait for all backends, then launch
WAIT_CMD=""
for pair in "${PAIRS[@]}"; do
    port="${pair##*:}"
    WAIT_CMD="${WAIT_CMD}while ! curl -s http://localhost:$port/v1/models >/dev/null 2>&1; do sleep 2; done; "
done

tmux send-keys -t "$PROXY_PANE" \
  "cd $SCRIPT_DIR && source .venv/bin/activate && \
   echo 'waiting for backends...' && $WAIT_CMD echo 'all backends up' && \
   ROUTES=$MODELS PROXY_PORT=$PROXY_PORT python3 route_proxy.py" C-m

tmux set-option -t "$SESSION" status-left-length 60
tmux set-option -t "$SESSION" status-left "  mlx-multi | $MODELS | proxy:$PROXY_PORT  "

echo ""
echo "  session: $SESSION"
echo "  endpoint for clients: http://localhost:$PROXY_PORT/v1"
echo "  backends: $MODELS"
echo "  detach:  Ctrl-b d"
echo "  kill:    ./start_tmux.sh --kill"
echo ""

if [ "${1:-}" != "--detach" ]; then
    sleep 1
    tmux attach -t "$SESSION"
fi
