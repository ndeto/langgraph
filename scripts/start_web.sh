#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
uv run alembic upgrade head

cd "$ROOT_DIR/frontend"
npm run build -- --outDir ../src/atlasai/web/static --emptyOutDir

cd "$ROOT_DIR"
exec uv run fastapi dev src/atlasai/web/main.py
