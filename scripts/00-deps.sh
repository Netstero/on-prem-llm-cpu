#!/usr/bin/env bash
# 00-deps.sh — install build deps for llama.cpp + sanity-check the CT for CPU inference.
# Prereqs: run ON the llm CT as a sudo-capable user. Idempotent (apt + checks safe to rerun).
# Usage: ./00-deps.sh   (or via install.sh). Values come from config.env (SSOT).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/config.env"

log(){ printf '[%s] %s\n' "$(date '+%F %H:%M:%S')" "$*"; }
die(){ printf '[%s] FATAL: %s\n' "$(date '+%F %H:%M:%S')" "$*" >&2; exit 1; }

log "Sanity checks…"
grep -q avx2 /proc/cpuinfo || die "CPU lacks AVX2 — llama.cpp would be unusably slow"
cores=$(nproc); log "nproc=$cores (config THREADS=$THREADS)"
[ "$cores" -ge "$THREADS" ] || log "WARN: nproc ($cores) < THREADS ($THREADS) — lower THREADS in config.env"
ramg=$(free -g | awk '/Mem:/{print $2}'); log "RAM=${ramg}G"
[ "$ramg" -ge 16 ] || die "RAM ${ramg}G < 16G minimum for the model"

log "Installing build deps (idempotent)…"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential git cmake libcurl4-openssl-dev libssl-dev pkg-config ca-certificates curl
log "Versions: gcc=$(gcc -dumpversion) cmake=$(cmake --version | head -1 | awk '{print $3}') git=$(git --version | awk '{print $3}')"
log "00-deps.sh DONE"
