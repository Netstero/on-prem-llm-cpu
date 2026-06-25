#!/usr/bin/env bash
# 20-fetch-model.sh — download the GPT-OSS GGUF into MODEL_DIR. Idempotent (resume/skip).
# Prereqs: curl (from 00-deps). Usage: ./20-fetch-model.sh (or via install.sh).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/config.env"

log(){ printf '[%s] %s\n' "$(date '+%F %H:%M:%S')" "$*"; }
die(){ printf '[%s] FATAL: %s\n' "$(date '+%F %H:%M:%S')" "$*" >&2; exit 1; }

url="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
dest="$MODEL_DIR/$MODEL_FILE"
MIN_BYTES=10000000000   # ~10 GB sanity floor (file is ~12 GB)

if [ ! -d "$MODEL_DIR" ]; then
  sudo mkdir -p "$MODEL_DIR"
  sudo chown "$(id -u):$(id -g)" "$MODEL_DIR"
fi

if [ -f "$dest" ]; then
  sz=$(stat -c%s "$dest")
  if [ "$sz" -gt "$MIN_BYTES" ]; then log "Model present ($((sz/1024/1024)) MB) — skip"; exit 0; fi
  log "Partial file ($((sz/1024/1024)) MB) — resuming"
fi

log "Downloading $url"
curl -L --fail --retry 5 --retry-delay 5 -C - -o "$dest" "$url"

sz=$(stat -c%s "$dest")
[ "$sz" -gt "$MIN_BYTES" ] || die "Downloaded file only $((sz/1024/1024)) MB (<10 GB) — aborting"
log "20-fetch-model.sh DONE → $dest ($((sz/1024/1024)) MB)"
