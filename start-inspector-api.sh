#!/usr/bin/env bash
# Authors: LCC Research Lab
# Project: From Document to Evidence
#
# Start the AI Model Inspector API on this Mac.
# Usage:
#   ./start-inspector-api.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ROOT}/inspector-engine"
PORT="${INSPECTOR_PORT:-8810}"
case "${INSPECTOR_HOST:-127.0.0.1}" in
  ""|127.0.0.1|localhost|::1) HOST="127.0.0.1" ;;
  *)
    echo "The Inspector API listens on 127.0.0.1 only." >&2
    exit 1
    ;;
esac

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need python3

cd "${ENGINE}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating the Inspector API environment"
  python3 -m venv .venv
fi

echo "Installing Inspector API packages"
.venv/bin/python -m pip install -q -r requirements.txt

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Inspector API is already listening on http://${HOST}:${PORT}"
  exit 0
fi

echo "Starting Inspector API at http://${HOST}:${PORT}"
export INSPECTOR_HOST="${HOST}"
export INSPECTOR_PORT="${PORT}"
exec .venv/bin/python -m uvicorn server:app --host "${HOST}" --port "${PORT}"
