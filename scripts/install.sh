#!/usr/bin/env bash
# install.sh — master runner: full GPT-OSS install on the CT, from scratch.
# Run ON the llm CT: bash ~/gpt-oss/scripts/install.sh
# Idempotent end-to-end (each step skips/repeats safely). Values from config.env (SSOT).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/config.env"
log(){ printf '[%s] %s\n' "$(date '+%F %H:%M:%S')" "$*"; }

log "=== GPT-OSS install START ($CT_HOST) ==="
bash "$here/00-deps.sh"
bash "$here/10-build-llamacpp.sh"
bash "$here/20-fetch-model.sh"
bash "$here/30-systemd.sh"
log "=== GPT-OSS install COMPLETE → http://$CT_IP:$PORT/v1 ==="
