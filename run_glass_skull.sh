#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${STREAMLIT_SERVER_PORT:-8501}"
ADDRESS="${STREAMLIT_SERVER_ADDRESS:-localhost}"
START_LLAMA_CPP="${START_LLAMA_CPP:-1}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-/home/dsmason321/models/Best/Qwen3.6-35B-MTP-Q4_KS.gguf}"
LLAMA_MODEL_ALIAS="${LLAMA_MODEL_ALIAS:-qwen3.6-35b-mtp-q4-ks-vision}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"
LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-$SCRIPT_DIR/data/logs/llama-server.log}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:---jinja --flash-attn auto --cache-type-k q4_0 --cache-type-v q4_0 --no-mmap}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-999}"
LLAMA_PID=""

on_exit() {
    status=$?
    if [[ -n "${LLAMA_PID:-}" ]] && kill -0 "$LLAMA_PID" >/dev/null 2>&1; then
        echo
        echo "Stopping llama.cpp server pid $LLAMA_PID"
        kill "$LLAMA_PID" >/dev/null 2>&1 || true
        wait "$LLAMA_PID" >/dev/null 2>&1 || true
    fi
    if [[ $status -ne 0 && $status -ne 130 && -t 0 ]]; then
        echo
        echo "Glass Skull stopped with exit code $status."
        read -r -p "Press Enter to close this window..."
    fi
    exit "$status"
}
trap on_exit EXIT

log() {
    echo
    echo "==> $*"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

port_is_open() {
    python3 - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
sock.settimeout(0.25)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

find_llama_server_bin() {
    if [[ -n "$LLAMA_SERVER_BIN" ]]; then
        echo "$LLAMA_SERVER_BIN"
        return
    fi
    if [[ -x /home/dsmason321/llama.cpp/build/bin/llama-server ]]; then
        echo /home/dsmason321/llama.cpp/build/bin/llama-server
        return
    fi
    if [[ -x "$SCRIPT_DIR/managed/llama.cpp-glass/build/bin/llama-server" ]]; then
        echo "$SCRIPT_DIR/managed/llama.cpp-glass/build/bin/llama-server"
        return
    fi
    echo "$SCRIPT_DIR/managed/llama.cpp-glass/build/bin/llama-server"
}

check_no_existing_llama() {
    if command -v pgrep >/dev/null 2>&1; then
        local existing
        existing="$(pgrep -af '(^|/)(llama-server|llama.cpp)( |$)' || true)"
        if [[ -n "$existing" ]]; then
            echo "Refusing to start llama.cpp because another llama.cpp process appears to be running:" >&2
            echo "$existing" >&2
            echo >&2
            echo "Stop that process first, or run with START_LLAMA_CPP=0 to launch only Glass Skull." >&2
            exit 1
        fi
    fi

    if port_is_open "$LLAMA_HOST" "$LLAMA_PORT"; then
        echo "Refusing to start llama.cpp because $LLAMA_HOST:$LLAMA_PORT is already accepting connections." >&2
        echo "Stop the existing server or change LLAMA_PORT." >&2
        exit 1
    fi
}

wait_for_llama() {
    local url="http://$LLAMA_HOST:$LLAMA_PORT/v1/models"
    local attempts=90
    for _ in $(seq 1 "$attempts"); do
        if ! kill -0 "$LLAMA_PID" >/dev/null 2>&1; then
            echo "llama.cpp server exited during startup. Last log lines:" >&2
            tail -80 "$LLAMA_LOG_FILE" >&2 || true
            exit 1
        fi
        if python3 - "$url" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=1.0) as resp:
        json.loads(resp.read().decode("utf-8"))
except Exception:
    sys.exit(1)
sys.exit(0)
PY
        then
            return 0
        fi
        sleep 2
    done

    echo "Timed out waiting for llama.cpp at $url. Last log lines:" >&2
    tail -80 "$LLAMA_LOG_FILE" >&2 || true
    exit 1
}

start_llama_cpp() {
    if [[ "$START_LLAMA_CPP" == "0" || "$START_LLAMA_CPP" == "false" ]]; then
        log "Skipping llama.cpp startup because START_LLAMA_CPP=$START_LLAMA_CPP"
        return
    fi

    local server_bin
    server_bin="$(find_llama_server_bin)"
    if [[ ! -x "$server_bin" ]]; then
        echo "llama-server binary not found or not executable: $server_bin" >&2
        echo "Build it first, or set LLAMA_SERVER_BIN=/path/to/llama-server." >&2
        exit 1
    fi
    if [[ ! -f "$LLAMA_MODEL_PATH" ]]; then
        echo "GGUF model path does not exist: $LLAMA_MODEL_PATH" >&2
        echo "Set LLAMA_MODEL_PATH=/path/to/model.gguf." >&2
        exit 1
    fi

    check_no_existing_llama

    mkdir -p "$(dirname "$LLAMA_LOG_FILE")"
    log "Starting llama.cpp on http://$LLAMA_HOST:$LLAMA_PORT"
    echo "llama.cpp log: $LLAMA_LOG_FILE"

    # shellcheck disable=SC2206
    local extra_args=( $LLAMA_EXTRA_ARGS )
    "$server_bin" \
        -m "$LLAMA_MODEL_PATH" \
        --host "$LLAMA_HOST" \
        --port "$LLAMA_PORT" \
        -ngl "$LLAMA_N_GPU_LAYERS" \
        --alias "$LLAMA_MODEL_ALIAS" \
        "${extra_args[@]}" \
        >"$LLAMA_LOG_FILE" 2>&1 &
    LLAMA_PID="$!"

    wait_for_llama
    log "llama.cpp is ready"
}

require_command python3

log "Starting Glass Skull from $SCRIPT_DIR"

if [[ ! -d .venv ]]; then
    log "Creating local Python environment"
    python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

log "Installing Python requirements"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if grep -q "def render_hf_catalog_panel" main.py && grep -q "workflow_setup_complete" main.py; then
    log "Local UI patch already present"
else
    log "Applying local UI patch"
    python scripts/apply_hf_front_patch.py
fi

log "Running startup checks"
python -m compileall main.py glass_skull scripts smoke_check.py
python smoke_check.py

start_llama_cpp

log "Launching Streamlit"
echo "Open this URL if the browser does not open automatically:"
echo "http://$ADDRESS:$PORT"
echo "llama.cpp: http://$LLAMA_HOST:$LLAMA_PORT"
echo

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
python -m streamlit run main.py --server.address "$ADDRESS" --server.port "$PORT"
