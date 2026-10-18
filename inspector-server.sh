#!/usr/bin/env bash
# Authors: LCC Research Lab
# Project: From Document to Evidence
#
# Manage the AI Model Inspector API on this Mac.
# Usage:
#   ./inspector-server.sh status|start|stop|restart [--port PORT]
#
# Port priority: --port, then INSPECTOR_PORT, then 8810.
# Each port has its own PID file and log.
#
# Exit codes:
#   0  success, including an API that is already running
#   1  stopped server, invalid input, or the API did not become healthy
#   2  port occupied by another process
#   3  an Inspector process is listening, but its health check failed

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ROOT}/inspector-engine"
RUNTIME="${ROOT}/.inspector-runtime"
HOST="127.0.0.1"
DEFAULT_PORT=8810
STOP_TIMEOUT=10
COMMAND=""
PORT_ARG=""
JSON=0

print_help() {
  cat <<EOF
Usage: ./inspector-server.sh status|start|stop|restart [--port PORT]

Port priority: --port, INSPECTOR_PORT, default ${DEFAULT_PORT}.
Each port has its own PID file and log in .inspector-runtime/.

Add --json for one JSON object on stdout. Other programs use that
result and ignore diagnostic text.

Exit codes:
  0  success, including an API that is already running
  1  stopped server, invalid input, or the API did not become healthy
  2  port occupied by another process
  3  an Inspector process is listening, but its health check failed
EOF
}

usage() {
  print_help >&2
  exit 1
}

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

listener_pids() {
  lsof -nP -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
}

process_cwd() {
  lsof -a -p "$1" -d cwd -Fn 2>/dev/null | awk '/^n/ { print substr($0, 2); exit }'
}

process_command() {
  ps -p "$1" -o command= 2>/dev/null || true
}

is_inspector_pid() {
  local pid="$1"
  local cmd cwd
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  cmd="$(process_command "$pid")"
  cwd="$(process_cwd "$pid")"
  [[ "$cmd" == *"-m uvicorn"* && "$cmd" == *"server:app"* && "$cmd" == *"--port ${PORT}"* ]] || return 1
  [[ "$cwd" == "$ENGINE" ]] || return 1
}

pid_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

health_json() {
  curl -sf --max-time 2 "http://${HOST}:${PORT}/health" 2>/dev/null || true
}

health_is_inspector() {
  local body
  body="$(health_json)"
  [[ -n "$body" ]] || return 1
  BODY="$body" EXPECTED_PORT="$PORT" python3 - <<'PY'
import json, os, sys
try:
    data = json.loads(os.environ["BODY"])
except json.JSONDecodeError:
    sys.exit(1)
try:
    expected = int(os.environ["EXPECTED_PORT"])
except ValueError:
    sys.exit(1)
if data.get("ok") is True and data.get("service") == "ai-model-inspector" and data.get("port") == expected:
    sys.exit(0)
sys.exit(1)
PY
}

clear_stale_pidfile() {
  local file pid
  for file in "$PIDFILE" "${RUNTIME}/inspector-api.pid"; do
    [[ -f "$file" ]] || continue
    pid="$(tr -d '[:space:]' <"$file")"
    if ! pid_alive "$pid"; then
      rm -f "$file"
      progress "Removed stale PID file for process ${pid}."
    fi
  done
}

ensure_runtime() {
  mkdir -p "$RUNTIME"
}

ensure_venv() {
  need python3
  if [[ ! -x "${ENGINE}/.venv/bin/python" ]]; then
    progress "Creating the Inspector API environment"
    python3 -m venv "${ENGINE}/.venv"
  fi
  progress "Installing Inspector API packages"
  "${ENGINE}/.venv/bin/python" -m pip install -q -r "${ENGINE}/requirements.txt"
}

progress() {
  if [[ "$JSON" -eq 1 ]]; then
    printf '%s\n' "$*" >&2
  else
    printf '%s\n' "$*"
  fi
}

emit() {
  local state="$1" code="$2" message="$3" pid="${4:-}"
  if [[ "$JSON" -eq 1 ]]; then
    STATE="$state" MESSAGE="$message" PID="$pid" PORT="$PORT" python3 - <<'PY'
import json, os
pid = os.environ.get("PID", "")
print(json.dumps({
    "state": os.environ["STATE"],
    "port": int(os.environ["PORT"]),
    "pid": int(pid) if pid.isdigit() else None,
    "message": os.environ["MESSAGE"],
}))
PY
  elif [[ "$code" -eq 0 ]]; then
    printf '%s\n' "$message"
  else
    printf '%s\n' "$message" >&2
  fi
  return "$code"
}

classify_listeners() {
  local pids pid
  INSPECTOR_PID_LIST=""
  FOREIGN_PID_LIST=""
  pids="$(listener_pids)"
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if is_inspector_pid "$pid"; then
      INSPECTOR_PID_LIST="${INSPECTOR_PID_LIST}${pid}"$'\n'
    else
      FOREIGN_PID_LIST="${FOREIGN_PID_LIST}${pid}"$'\n'
    fi
  done <<<"$pids"
}

first_pid() {
  printf '%s\n' "$1" | awk 'NF { print; exit }'
}

wait_until_healthy() {
  local attempt
  for attempt in $(seq 1 50); do
    if health_is_inspector; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

describe_foreign() {
  local pid
  if [[ "$JSON" -eq 1 ]]; then
    return 0
  fi
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    echo "PID ${pid}: $(process_command "$pid")" >&2
  done <<<"$FOREIGN_PID_LIST"
}

cmd_status() {
  local pid
  clear_stale_pidfile
  classify_listeners
  pid="$(first_pid "$INSPECTOR_PID_LIST")"
  if health_is_inspector; then
    if [[ -n "$FOREIGN_PID_LIST" || -z "$pid" ]]; then
      describe_foreign
      emit foreign 2 "Port ${PORT} answered, but the Inspector process identity was not confirmed. Nothing was stopped." || return $?
      return 2
    fi
    emit running 0 "Inspector API is running at http://${HOST}:${PORT} (PID ${pid})" "$pid" || return $?
    return 0
  fi
  if [[ -z "$INSPECTOR_PID_LIST" && -z "$FOREIGN_PID_LIST" ]]; then
    emit stopped 1 "Inspector API is not running." || return $?
    return 1
  fi
  if [[ -n "$pid" && -z "$FOREIGN_PID_LIST" ]]; then
    emit unresponsive 3 "The Inspector process is listening on port ${PORT}, but the health check failed." "$pid" || return $?
    return 3
  fi
  describe_foreign
  emit foreign 2 "Port ${PORT} is in use, and it is not the AI Model Inspector API." || return $?
  return 2
}

cmd_start() {
  local pid old
  clear_stale_pidfile
  classify_listeners
  pid="$(first_pid "$INSPECTOR_PID_LIST")"
  if health_is_inspector && [[ -n "$pid" && -z "$FOREIGN_PID_LIST" ]]; then
    emit running 0 "Inspector API is already running at http://${HOST}:${PORT} (PID ${pid})" "$pid" || return $?
    return 0
  fi
  if [[ -n "$FOREIGN_PID_LIST" ]]; then
    describe_foreign
    emit foreign 2 "Port ${PORT} is already used by another process. It was not stopped." || return $?
    return 2
  fi
  if [[ -n "$INSPECTOR_PID_LIST" ]]; then
    emit unresponsive 3 "The Inspector process is already listening on port ${PORT}, but it is not healthy." "$pid" || return $?
    return 3
  fi
  if [[ -f "$PIDFILE" ]]; then
    old="$(tr -d '[:space:]' <"$PIDFILE")"
    if pid_alive "$old" && ! is_inspector_pid "$old"; then
      emit foreign 2 "PID file points at a live process that is not the Inspector API. It was not stopped." || return $?
      return 2
    fi
  fi
  ensure_runtime
  ensure_venv
  progress "Starting Inspector API at http://${HOST}:${PORT}"
  echo "---- $(date -u '+%Y-%m-%dT%H:%M:%SZ') start ----" >>"$LOGFILE"
  pid="$("${ENGINE}/.venv/bin/python" -c '
import os, sys
engine, logpath, host, port = sys.argv[1:5]
pid = os.fork()
if pid > 0:
    print(pid)
    raise SystemExit(0)
os.setsid()
os.chdir(engine)
os.environ["INSPECTOR_HOST"] = host
os.environ["INSPECTOR_PORT"] = port
log = open(logpath, "a", buffering=1)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
devnull = os.open(os.devnull, os.O_RDONLY)
os.dup2(devnull, 0)
python = os.path.join(engine, ".venv", "bin", "python")
os.execv(python, [python, "-m", "uvicorn", "server:app", "--host", host, "--port", port])
' "$ENGINE" "$LOGFILE" "$HOST" "$PORT")"
  printf '%s\n' "$pid" >"$PIDFILE"
  if wait_until_healthy; then
    pid="$(tr -d '[:space:]' <"$PIDFILE")"
    emit running 0 "Inspector API is running at http://${HOST}:${PORT} (PID ${pid})" "$pid" || return $?
    return 0
  fi
  if [[ "$JSON" -eq 0 ]]; then
    echo "Inspector API did not become healthy. Recent log:" >&2
    tail -n 40 "$LOGFILE" >&2 || true
  fi
  emit error 1 "Inspector API did not become healthy." || return $?
  return 1
}

stop_verified_pid() {
  local pid="$1"
  local attempt
  if ! pid_alive "$pid"; then
    return 0
  fi
  if ! is_inspector_pid "$pid"; then
    echo "Refusing to stop PID ${pid}. It is not the Inspector API." >&2
    return 2
  fi
  kill -TERM "$pid" 2>/dev/null || true
  for attempt in $(seq 1 "$STOP_TIMEOUT"); do
    if ! pid_alive "$pid"; then
      return 0
    fi
    sleep 1
  done
  if pid_alive "$pid" && is_inspector_pid "$pid"; then
    kill -KILL "$pid" 2>/dev/null || true
    sleep 0.2
  fi
  if pid_alive "$pid"; then
    echo "PID ${pid} did not exit." >&2
    return 1
  fi
  return 0
}

cmd_stop() {
  local pid stop_code
  clear_stale_pidfile
  classify_listeners
  if [[ -z "$INSPECTOR_PID_LIST" && -z "$FOREIGN_PID_LIST" ]]; then
    rm -f "$PIDFILE"
    emit stopped 0 "Inspector API is not running." || return $?
    return 0
  fi
  if [[ -n "$FOREIGN_PID_LIST" ]]; then
    describe_foreign
    emit foreign 2 "Port ${PORT} is in use by another application. It was not stopped." || return $?
    return 2
  fi
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    stop_verified_pid "$pid" && continue
    stop_code=$?
    emit error "$stop_code" "The Inspector process was not stopped." || return $?
    return "$stop_code"
  done <<<"$INSPECTOR_PID_LIST"
  rm -f "$PIDFILE"
  emit stopped 0 "Inspector API stopped." || return $?
  return 0
}

cmd_restart() {
  cmd_stop
  cmd_start
}

need curl
need lsof
need python3

while [[ $# -gt 0 ]]; do
  case "$1" in
    status|start|stop|restart)
      if [[ -n "$COMMAND" ]]; then
        usage
      fi
      COMMAND="$1"
      shift
      ;;
    --port)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Missing value for --port." >&2
        exit 1
      fi
      PORT_ARG="$2"
      shift 2
      ;;
    --port=*)
      PORT_ARG="${1#*=}"
      if [[ -z "$PORT_ARG" ]]; then
        echo "Missing value for --port." >&2
        exit 1
      fi
      shift
      ;;
    --json)
      JSON=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "$COMMAND" ]]; then
  usage
fi

if [[ -n "$PORT_ARG" ]]; then
  PORT="$PORT_ARG"
elif [[ -n "${INSPECTOR_PORT:-}" ]]; then
  PORT="$INSPECTOR_PORT"
else
  PORT="$DEFAULT_PORT"
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( 10#$PORT < 1 || 10#$PORT > 65535 )); then
  echo "Port must be an integer from 1 to 65535." >&2
  exit 1
fi

case "${INSPECTOR_HOST:-127.0.0.1}" in
  ""|127.0.0.1|localhost|::1) HOST="127.0.0.1" ;;
  *)
    echo "The Inspector API listens on 127.0.0.1 only." >&2
    exit 1
    ;;
esac

PIDFILE="${RUNTIME}/inspector-api-${PORT}.pid"
LOGFILE="${RUNTIME}/inspector-api-${PORT}.log"

if [[ "$PORT" -eq 8787 || "$PORT" -eq 8799 || "$PORT" -eq 8811 ]]; then
  emit foreign 2 "Port ${PORT} is reserved for another local service. Nothing was changed." || exit $?
  exit 2
fi

case "$COMMAND" in
  status) cmd_status ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_restart ;;
  *) usage ;;
esac
