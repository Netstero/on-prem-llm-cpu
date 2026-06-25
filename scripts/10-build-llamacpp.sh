#!/usr/bin/env bash
# 10-build-llamacpp.sh — clone & build llama.cpp (CPU-only) on the CT.
# Prereqs: 00-deps.sh done. Idempotent: re-fetches if repo exists, rebuilds incrementally.
# Usage: ./10-build-llamacpp.sh   (or via install.sh). Values from config.env (SSOT).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/config.env"

log(){ printf '[%s] %s\n' "$(date '+%F %H:%M:%S')" "$*"; }
die(){ printf '[%s] FATAL: %s\n' "$(date '+%F %H:%M:%S')" "$*" >&2; exit 1; }

if [ ! -d "$LLAMA_DIR/.git" ]; then
  log "Cloning $LLAMA_REPO → $LLAMA_DIR"
  sudo mkdir -p "$LLAMA_DIR"
  sudo chown "$(id -u):$(id -g)" "$LLAMA_DIR"
  git clone "$LLAMA_REPO" "$LLAMA_DIR"
else
  log "Repo present; fetching latest"
  git -C "$LLAMA_DIR" fetch -q origin
fi

if [ -n "${LLAMA_COMMIT:-}" ]; then
  log "Checking out pinned commit $LLAMA_COMMIT"
  git -C "$LLAMA_DIR" checkout -q "$LLAMA_COMMIT"
else
  git -C "$LLAMA_DIR" checkout -q origin/HEAD 2>/dev/null || git -C "$LLAMA_DIR" pull -q --ff-only || true
fi
sha=$(git -C "$LLAMA_DIR" rev-parse --short HEAD)

log "Configuring (Release, CURL on) @ $sha"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON >/dev/null

log "Building -j$BUILD_JOBS (this takes a few minutes)…"
cmake --build "$LLAMA_DIR/build" --config Release -j"$BUILD_JOBS"

test -x "$LLAMA_DIR/build/bin/llama-server" || die "llama-server missing after build"
test -x "$LLAMA_DIR/build/bin/llama-cli"    || die "llama-cli missing after build"
log "Built OK. Pin this for reproducible rebuilds → LLAMA_COMMIT=$sha"
log "10-build-llamacpp.sh DONE (commit $sha)"
