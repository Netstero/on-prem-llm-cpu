#!/usr/bin/env bash
# 30-systemd.sh — install & start gpt-oss.service (llama-server, OpenAI-compatible API).
# Prereqs: 10-build + 20-fetch done. Idempotent (overwrites unit, restarts).
# Usage: ./30-systemd.sh (or via install.sh). Values from config.env (SSOT).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/config.env"

log(){ printf '[%s] %s\n' "$(date '+%F %H:%M:%S')" "$*"; }
die(){ printf '[%s] FATAL: %s\n' "$(date '+%F %H:%M:%S')" "$*" >&2; exit 1; }

bin="$SERVE_BIN"

# Select active model + its sampling via the ACTIVE_MODEL toggle (SSOT in config.env).
case "${ACTIVE_MODEL:-gptoss}" in
  qwen)
    desc="Qwen3-Coder-30B-A3B (llama-server, CPU)"
    model="$MODEL_DIR/$QWEN_MODEL_FILE"
    temp="$QWEN_TEMP"; top_p="$QWEN_TOP_P"; top_k="$QWEN_TOP_K"; min_p="$QWEN_MIN_P"
    rep="--repeat-penalty $QWEN_REPEAT_PENALTY"
    ;;
  gptoss|*)
    desc="GPT-OSS 20B (llama-server, CPU)"
    model="$MODEL_DIR/$MODEL_FILE"
    temp="$TEMP"; top_p="$TOP_P"; top_k="$TOP_K"; min_p="$MIN_P"
    rep=""
    ;;
esac

test -x "$bin"   || die "serve binary missing ($bin) — build it (10-build-llamacpp.sh / ik build)"
test -f "$model" || die "model missing ($model) — run the fetch step"
log "ACTIVE_MODEL=${ACTIVE_MODEL:-gptoss} → serving $model"

unit=/etc/systemd/system/gpt-oss.service
log "Writing $unit"
sudo tee "$unit" >/dev/null <<EOF
[Unit]
Description=$desc
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$CT_USER
ExecStart=$bin -m $model --host $SERVE_HOST --port $PORT --ctx-size $CTX --parallel $SLOTS --jinja $FA --threads $THREADS --threads-batch $THREADS --temp $temp --top-p $top_p --top-k $top_k --min-p $min_p $rep
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gpt-oss
sudo systemctl restart gpt-oss

log "Waiting for model load + health…"
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "Server healthy on :$PORT after ~$((i*5))s"
    log "30-systemd.sh DONE"
    exit 0
  fi
  sleep 5
done
die "Server not healthy in 300s — check: journalctl -u gpt-oss -n 50"
