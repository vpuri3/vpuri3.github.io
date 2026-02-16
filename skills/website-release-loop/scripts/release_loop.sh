#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PID_FILE="${ROOT_DIR}/.hugo_server.pid"
LOG_FILE="${ROOT_DIR}/.hugo_server.log"

matching_pids() {
  pgrep -f "hugo server --bind 127.0.0.1 --port" || true
}

usage() {
  cat <<USAGE
Usage:
  $0 check
  $0 preview [port]
  $0 start [port]
  $0 stop
  $0 restart [port]
  $0 status
  $0 logs
  $0 publish [commit_message]

Commands:
  check    Run clean Hugo build
  preview  Start Hugo server in foreground on localhost (default port 1313)
  start    Start Hugo server in background and keep it alive for manual editing
  stop     Stop background Hugo server
  restart  Restart background Hugo server
  status   Show background server status
  logs     Show background server logs
  publish  Commit all tracked/untracked changes and push to origin/master
USAGE
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: '$1' is required but not installed." >&2
    exit 1
  }
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_bg() {
  require_cmd hugo
  local port="${1:-1313}"

  if is_running; then
    echo "Hugo server is already running (PID $(cat "$PID_FILE"))."
    echo "URL: http://localhost:${port}/"
    return 0
  fi

  # Clean stale matching processes from older launch modes.
  local stale
  stale="$(matching_pids)"
  if [[ -n "$stale" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      kill "$pid" 2>/dev/null || true
    done <<<"$stale"
    sleep 1
  fi

  (
    cd "$ROOT_DIR"
    nohup hugo server --bind 127.0.0.1 --port "$port" --disableFastRender >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
  )

  sleep 1
  if is_running; then
    echo "Started Hugo server in background."
    echo "PID: $(cat "$PID_FILE")"
    echo "URL: http://localhost:${port}/"
    echo "Log: $LOG_FILE"
  else
    echo "Failed to start Hugo server. See log: $LOG_FILE" >&2
    exit 1
  fi
}

stop_bg() {
  local did_stop=0
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid" 2>/dev/null || true
    did_stop=1
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi

  local others
  others="$(matching_pids)"
  if [[ -n "$others" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      kill "$pid" 2>/dev/null || true
      did_stop=1
    done <<<"$others"
  fi

  rm -f "$PID_FILE"
  if [[ "$did_stop" -eq 1 ]]; then
    echo "Stopped Hugo server process(es)."
  else
    echo "Hugo server is not running."
  fi
}

status_bg() {
  if is_running; then
    echo "Hugo server is running (PID $(cat "$PID_FILE"))."
    echo "URL: http://localhost:1313/"
    echo "Log: $LOG_FILE"
  else
    echo "Hugo server is not running."
  fi
}

cmd="${1:-}"

case "$cmd" in
  check)
    require_cmd hugo
    hugo --cleanDestinationDir
    ;;

  preview)
    require_cmd hugo
    port="${2:-1313}"
    hugo server -D --bind 127.0.0.1 --port "$port"
    ;;

  start)
    port="${2:-1313}"
    start_bg "$port"
    ;;

  stop)
    stop_bg
    ;;

  restart)
    port="${2:-1313}"
    stop_bg
    start_bg "$port"
    ;;

  status)
    status_bg
    ;;

  logs)
    touch "$LOG_FILE"
    tail -n 120 "$LOG_FILE"
    ;;

  publish)
    require_cmd git
    msg="${2:-Website update}"

    git add -A
    if git diff --cached --quiet; then
      echo "No changes staged; nothing to publish."
      exit 0
    fi

    git commit -m "$msg"
    git push origin master
    ;;

  *)
    usage
    exit 1
    ;;
esac
