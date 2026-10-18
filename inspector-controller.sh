#!/usr/bin/env bash
# Authors: LCC Research Lab
# Project: From Document to Evidence
#
# Keep a local controller available when the Inspector API is stopped.
# Usage: ./inspector-controller.sh status|start|stop
#
# The controller listens on 127.0.0.1:8811 only. It does not accept arbitrary
# commands. Exit 0 means success, 1 means stopped or a failed start, and 2
# means another process is using the controller port.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ROOT}/inspector-engine"
RUNTIME="${ROOT}/.inspector-runtime"
HOST="127.0.0.1"
PORT="8811"
PIDFILE="${RUNTIME}/inspector-controller.pid"
LOGFILE="${RUNTIME}/inspector-controller.log"
COMMAND=""

usage() {
  echo "Usage: ./inspector-controller.sh status|start|stop" >&2
  exit 1
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

is_controller_pid() {
  local pid="$1" cmd cwd
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  cmd="$(process_command "$pid")"
  cwd="$(process_cwd "$pid")"
  [[ "$cmd" == *"inspector_engine.server_controller"* && "$cmd" == *"--port ${PORT}"* ]] || return 1
  [[ "$cwd" == "$ENGINE" ]] || return 1
}

pid_alive() {
  [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null
}

health_ok() {
  curl -sf --max-time 2 "http://${HOST}:${PORT}/health" | python3 -c 'import json,sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
sys.exit(0 if data.get("ok") is True and data.get("service") == "ai-model-inspector-controller" else 1)'
}

cmd_status() {
  local pids pid found=0
  pids="$(listener_pids)"
  if [[ -z "$pids" ]]; then
    echo "Server controller is not running."
    return 1
  fi
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if is_controller_pid "$pid"; then
      echo "Server controller is running at http://${HOST}:${PORT} (PID ${pid})"
      found=1
    fi
  done <<<"$pids"
  if [[ "$found" -eq 1 ]]; then
    return 0
  fi
  echo "Port ${PORT} is in use, and it is not the AI Model Inspector controller." >&2
  return 2
}

cmd_start() {
  local pids pid
  if health_ok && cmd_status >/dev/null; then
    cmd_status
    return 0
  fi
  pids="$(listener_pids)"
  if [[ -n "$pids" ]]; then
    echo "Port ${PORT} is already used. The controller was not started." >&2
    return 2
  fi
  mkdir -p "$RUNTIME"
  echo "---- $(date -u '+%Y-%m-%dT%H:%M:%SZ') start ----" >>"$LOGFILE"
  pid="$(python3 -c '
import os, sys
engine, logpath = sys.argv[1:3]
pid = os.fork()
if pid > 0:
    print(pid)
    raise SystemExit(0)
os.setsid()
os.chdir(engine)
log = open(logpath, "a", buffering=1)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
devnull = os.open(os.devnull, os.O_RDONLY)
os.dup2(devnull, 0)
os.execvp(sys.executable, [sys.executable, "-m", "inspector_engine.server_controller", "--host", "127.0.0.1", "--port", "8811"])
' "$ENGINE" "$LOGFILE")"
  printf '%s\n' "$pid" >"$PIDFILE"
  local attempt
  for attempt in $(seq 1 25); do
    if health_ok; then
      echo "Server controller is running at http://${HOST}:${PORT} (PID ${pid})"
      return 0
    fi
    sleep 0.2
  done
  echo "Server controller did not become healthy." >&2
  return 1
}

cmd_stop() {
  local pids pid other=0
  pids="$(listener_pids)"
  if [[ -z "$pids" ]]; then
    rm -f "$PIDFILE"
    echo "Server controller is not running."
    return 0
  fi
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if ! is_controller_pid "$pid"; then
      other=1
    fi
  done <<<"$pids"
  if [[ "$other" -eq 1 ]]; then
    echo "Port ${PORT} is in use by another application. It was not stopped." >&2
    return 2
  fi
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done <<<"$pids"
  local attempt
  for attempt in $(seq 1 20); do
    if [[ -z "$(listener_pids)" ]]; then
      rm -f "$PIDFILE"
      echo "Server controller stopped."
      return 0
    fi
    sleep 0.25
  done
  echo "Server controller did not exit." >&2
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    status|start|stop)
      if [[ -n "$COMMAND" ]]; then usage; fi
      COMMAND="$1"
      shift
      ;;
    -h|--help)
      usage
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

case "$COMMAND" in
  status) cmd_status ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
esac
