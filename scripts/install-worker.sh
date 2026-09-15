#!/bin/bash
# Installs (or reinstalls) the worker launchd agent. Idempotent.
set -euo pipefail
LABEL=ru.slesarev.transcriber.worker
SRC="$(cd "$(dirname "$0")/.." && pwd)/launchd/$LABEL.plist"
DST="$HOME/Library/LaunchAgents/$LABEL.plist"
(cd "$(dirname "$SRC")/../worker" && uv sync --reinstall-package transcriber-worker >/dev/null)
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
cp "$SRC" "$DST"
launchctl bootstrap "gui/$(id -u)" "$DST"
launchctl kickstart -k "gui/$(id -u)/$LABEL"
echo "installed $LABEL; log: $HOME/Claude/transcriber/worker.log"
