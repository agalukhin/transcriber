#!/bin/bash
# Builds Transcriber.app from the SwiftPM package, signs it with the self-signed "Transcriber Dev"
# identity (stable designated requirement → TCC grants survive rebuilds), installs it to
# ~/Applications and (re)loads the launchd agent. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_SRC="$ROOT/app"
OUT="$ROOT/build/Transcriber.app"
DEST="$HOME/Applications/Transcriber.app"
LABEL=ru.slesarev.transcriber.app
IDENTITY="Transcriber Dev"

(cd "$APP_SRC" && swift build -c release --product Transcriber 2>&1 | grep -E "error|Build complete" || true)
BIN="$APP_SRC/.build/release/Transcriber"
[ -x "$BIN" ] || { echo "build failed: $BIN missing" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
cp "$BIN" "$OUT/Contents/MacOS/Transcriber"
cp "$APP_SRC/Resources/Info.plist" "$OUT/Contents/Info.plist"
codesign --force --sign "$IDENTITY" --timestamp=none "$OUT" || { echo "codesign failed (a keychain dialog for the Transcriber Dev key may be waiting)" >&2; exit 1; }
codesign --verify --verbose=2 "$OUT" || { echo "codesign verify failed" >&2; exit 1; }

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
pkill -x Transcriber 2>/dev/null || true
mkdir -p "$HOME/Applications" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
rm -rf "$DEST"
cp -R "$OUT" "$DEST"
cp "$ROOT/launchd/$LABEL.plist" "$HOME/Library/LaunchAgents/$LABEL.plist"
for i in 1 2 3 4 5; do
  launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/$LABEL.plist" && break
  sleep 1
done
echo "installed $DEST; agent $LABEL; log ~/Library/Logs/transcriber-app.log"
