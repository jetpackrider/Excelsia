#!/usr/bin/env bash
# Put the Discord bot on Fly.io, or update it. Run from anywhere:  bash scripts/fly.sh
#
# The bot's token is asked for once, typed blind, and sent straight to Fly.
# It is never written to a file here.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v fly > /dev/null || { echo "Install Fly's tool first:  brew install flyctl"; exit 1; }
[ -f bot/excelsia.npz ] || { echo "Pack her first:  python -m excelsia.export"; exit 1; }
fly auth whoami > /dev/null 2>&1 || fly auth login

if [ ! -f .fly-app ]; then
  read -r -p "Pick a name for the app on Fly (lower-case letters, numbers, dashes): " APP
  fly apps create "$APP"
  echo "$APP" > .fly-app
fi
APP="$(cat .fly-app)"

if ! fly secrets list -a "$APP" 2> /dev/null | grep -q DISCORD_TOKEN; then
  read -r -s -p "Paste the bot's token from Discord (nothing will show), then Enter: " TOKEN; echo
  read -r -p "Paste the id of her channel, then Enter: " CHANNEL
  printf 'DISCORD_TOKEN=%s\nEXCELSIA_CHANNEL=%s\n' "$TOKEN" "$CHANNEL" | fly secrets import -a "$APP"
  unset TOKEN
fi

fly deploy -a "$APP" --ha=false
echo
echo "Done. To watch her wake up:  fly logs -a $APP"
