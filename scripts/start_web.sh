#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTAPI_APP="${ATLASAI_FASTAPI_APP:-src/atlasai/web/main.py}"
FASTAPI_HOST="${ATLASAI_FASTAPI_HOST:-127.0.0.1}"
FASTAPI_PORT="${ATLASAI_FASTAPI_PORT:-8000}"
WORKER_CMD="${ATLASAI_WORKER_CMD:-uv run python -m atlasai.infrastructure.worker}"

cleanup() {
    local exit_code=$?

    if [[ -n "${FASTAPI_PID:-}" ]] && kill -0 "$FASTAPI_PID" 2>/dev/null; then
        kill "$FASTAPI_PID" 2>/dev/null || true
    fi

    if [[ -n "${WORKER_PID:-}" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
        kill "$WORKER_PID" 2>/dev/null || true
    fi

    wait 2>/dev/null || true
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
uv run alembic upgrade head

cd "$ROOT_DIR/frontend"
npm run build -- --outDir ../src/atlasai/web/static --emptyOutDir

cd "$ROOT_DIR"
uv run fastapi dev "$FASTAPI_APP" --host "$FASTAPI_HOST" --port "$FASTAPI_PORT" &
FASTAPI_PID=$!

eval "$WORKER_CMD" &
WORKER_PID=$!

while true; do
    if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
        wait "$FASTAPI_PID" 2>/dev/null || true
        break
    fi

    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        wait "$WORKER_PID" 2>/dev/null || true
        break
    fi

    sleep 1
done
