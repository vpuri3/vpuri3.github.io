#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 check
  $0 preview [port]
  $0 publish [commit_message]

Commands:
  check    Run clean Hugo build
  preview  Start Hugo server on localhost (default port 1313)
  publish  Commit all tracked/untracked changes and push to origin/master
USAGE
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: '$1' is required but not installed." >&2
    exit 1
  }
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
