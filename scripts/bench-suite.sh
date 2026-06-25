#!/usr/bin/env bash
# bench-suite.sh — full benchmark grid for the gpt-oss case study. Runs ON the CT.
#
# Methodology (this is what makes the numbers credible):
#   per measured run -> FRESH server process (cold KV + cold prompt cache)
#                    -> WARM-UP with a DIFFERENT throwaway prompt (can't be cached as the test)
#                    -> MEASURE via curl to 127.0.0.1 (no SSH in the request path)
#                    -> SAVE raw request + response (content + reasoning + usage + timings)
#                    -> VERIFY correctness (task-specific) -> append results.csv -> kill server.
# Engines switched by launching each engine's binary directly per cell (cleanest isolation;
# the live gpt-oss.service is stopped for the duration and restored at the end).
#
# Self-defending (no CT reboot needed — a fresh PROCESS per run is the real isolation):
#   - PREFLIGHT clean-slate gate: aborts the whole suite if the box isn't clean (stray server,
#     port busy, or <15GB RAM) instead of burning the night on biased/empty data.
#   - Per run: stop_server (SIGTERM→SIGKILL) before AND after; assert exactly ONE server instance
#     (clean+retry once, else FLAG); memory-headroom check; cold-cache bias detector.
#   - Every suspicious row gets a `flags` value; the end-of-run "Data health" summary lists them.
# Idempotent/resumable: skip keys on the CSV ROW (written last), so an interrupted run re-does
# cleanly — re-launch with the SAME RUN_ID to continue (even across a reboot).
#
# Launch detached:  mkdir -p ~/gpt-oss/results
#   nohup bash ~/gpt-oss/scripts/bench-suite.sh >> ~/gpt-oss/results/suite.out 2>&1 &
# Smoke test (fast, ik only, smallest input, 1 rep): SMOKE=1 bash ~/gpt-oss/scripts/bench-suite.sh
# PAUSE:            pkill -f bench-suite.sh ; pkill -f 'llama-server.*--port 8080'
#                   (leaves gpt-oss.service stopped — `sudo systemctl start gpt-oss` to serve meanwhile)
# RESUME:           RUN_ID=<id> nohup bash ~/gpt-oss/scripts/bench-suite.sh >> ~/gpt-oss/results/suite.out 2>&1 &
#                   (skips every cell-rep already in results.csv; finished work is never repeated)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/config.env"

API="127.0.0.1:$PORT"
REPS="${REPS:-3}"
OUT_SHORT="${OUT_SHORT:-64}"
OUT_LONG="${OUT_LONG:-2000}"
RESULTS_BASE="${RESULTS_BASE:-$HOME/gpt-oss/results}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
RDIR="$RESULTS_BASE/$RUN_ID"
CSV="$RDIR/results.csv"
SRVLOG="$RDIR/server.log"
PIDFILE="$RDIR/server.pid"
mkdir -p "$RDIR/raw"

# Active model + its sampling (ACTIVE_MODEL toggle in config.env; mirrors 30-systemd.sh).
# Held constant for the whole grid; engines (ik/main/ikfa) vary per cell, the model does not.
case "${ACTIVE_MODEL:-gptoss}" in
  qwen)     MODEL_TAG=qwen;   MODELF="$QWEN_MODEL_FILE"; S_TEMP="$QWEN_TEMP"; S_TOPP="$QWEN_TOP_P"; S_TOPK="$QWEN_TOP_K"; S_MINP="$QWEN_MIN_P"; S_REP="--repeat-penalty $QWEN_REPEAT_PENALTY";;
  gptoss|*) MODEL_TAG=gptoss; MODELF="$MODEL_FILE";      S_TEMP="$TEMP";      S_TOPP="$TOP_P";      S_TOPK="$TOP_K";      S_MINP="$MIN_P";      S_REP="";;
esac

log(){ printf '[%s] %s\n' "$(date '+%F %H:%M:%S')" "$*"; }

bin_for(){ case "$1" in
  ik|ikfa) echo "$IK_DIR/build/bin/llama-server";;
  main)    echo "$LLAMA_DIR/build/bin/llama-server";;
  *)       echo "";; esac; }
fa_for(){ case "$1" in ikfa) echo "-fa 1";; *) echo "";; esac; }   # flash-attn only for the *fa variant

# pgrep -c prints "0" AND exits 1 on no match, so `|| echo 0` would double it → capture stdout only.
count_servers(){ local n; n=$(pgrep -fc "llama-server.*--port $PORT" 2>/dev/null); printf '%s' "${n:-0}"; }
avail_mb(){ free -m | awk '/^Mem:/{print $7}'; }

stop_server(){   # TERM, wait, then escalate to KILL; verify gone + port free
  [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null
  pkill -TERM -f "llama-server.*--port $PORT" 2>/dev/null
  for _ in $(seq 1 20); do [ "$(count_servers)" -eq 0 ] && break; sleep 1; done
  if [ "$(count_servers)" -ne 0 ]; then
    log "WARN: server ignored SIGTERM — escalating to SIGKILL"
    pkill -KILL -f "llama-server.*--port $PORT" 2>/dev/null; sleep 2
  fi
  for _ in $(seq 1 30); do curl -fsS "http://$API/health" >/dev/null 2>&1 || { sleep 1; return 0; }; sleep 1; done
  log "WARN: port $PORT still answering after kill"
}

# Clean-slate gate, run ONCE before the grid (and on every resume). Refuses to start on a dirty box
# rather than burning the night on biased/empty data. Returns 1 => caller aborts.
preflight(){
  log "PREFLIGHT: stopping service + killing any stray llama-server"
  sudo systemctl stop gpt-oss 2>/dev/null || true
  pkill -KILL -f "llama-server.*--port $PORT" 2>/dev/null; sleep 2
  local n a; n="$(count_servers)"; a="$(avail_mb)"
  if [ "$n" -ne 0 ]; then log "ABORT preflight: $n llama-server still running after SIGKILL"; return 1; fi
  if curl -fsS "http://$API/health" >/dev/null 2>&1; then log "ABORT preflight: something still serving on $PORT"; return 1; fi
  if [ "${a:-0}" -lt 15000 ]; then log "ABORT preflight: only ${a}MB RAM available (<15000) — box not clean"; return 1; fi
  log "PREFLIGHT OK: 0 servers, port free, ${a}MB available"
  return 0
}

start_server(){ # $1=binary  $2=flash-attn flag ("" or "-fa 1")
  : > "$SRVLOG"
  nohup "$1" -m "$MODEL_DIR/$MODELF" --host 127.0.0.1 --port "$PORT" \
    --ctx-size "$CTX" --parallel "$SLOTS" --jinja $2 \
    --threads "$THREADS" --threads-batch "$THREADS" \
    --temp "$S_TEMP" --top-p "$S_TOPP" --top-k "$S_TOPK" --min-p "$S_MINP" $S_REP \
    >> "$SRVLOG" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 120); do curl -fsS "http://$API/health" >/dev/null 2>&1 && return 0; sleep 5; done
  log "ERROR: server ($1) not healthy in 600s — see $SRVLOG"; return 1
}

warmup(){
  curl -s --max-time 120 "http://$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with the single word: ready."}],"max_tokens":8,"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0}' \
    >/dev/null 2>&1 || true
}

# run one measured cell-rep: $1 engine $2 cell $3 task $4 sections $5 maxtok $6 rep
# Measures COLD (fresh server, cold cache) then immediately re-sends the IDENTICAL prompt with
# max_tokens=16 WITHOUT restarting → WARM (prefix cache hit) → records warm prefill/TTFT too.
run(){
  local engine="$1" cell="$2" task="$3" sections="$4" maxtok="$5" rep="$6"
  local base="$RDIR/raw/${cell}_rep${rep}"
  local req="${base}.req.json" resp="${base}.resp.json"
  local wreq="${base}.warm.req.json" wresp="${base}.warm.resp.json"
  # Resume-safe skip: key on the CSV row (written LAST), not the raw file (a killed run can leave a
  # truncated .resp.json). A run interrupted before its row is appended simply re-runs and overwrites.
  if [ -f "$CSV" ] && grep -q ",${cell},${task},${rep}," "$CSV" 2>/dev/null; then
    log "SKIP $cell rep$rep (already in csv)"; return 0; fi
  local bin fa; bin="$(bin_for "$engine")"; fa="$(fa_for "$engine")"
  [ -x "$bin" ] || { log "FAIL $cell rep$rep: binary missing ($bin)"; return 1; }
  log "RUN  $cell rep$rep  (task=$task sections=$sections max_tokens=$maxtok fa='${fa:-off}')"
  stop_server                                  # guarantee a clean slate even if a prior run left junk
  start_server "$bin" "$fa" || { log "FAIL start $cell rep$rep"; return 1; }
  warmup
  # --- GUARD: exactly one server (contention = biased numbers); clean+retry once, else flag ---
  local flag="" inst; inst="$(count_servers)"
  if [ "$inst" -ne 1 ]; then
    log "WARN $cell rep$rep: $inst server instances (want 1) — cleaning + retrying once"
    stop_server; start_server "$bin" "$fa" || { log "FAIL restart $cell rep$rep"; return 1; }
    warmup; inst="$(count_servers)"
    [ "$inst" -ne 1 ] && { flag="multi_instance($inst);"; log "WARN $cell rep$rep: STILL $inst instances — flagged"; }
  fi
  # --- GUARD: memory headroom (model ~12GB; flag if little left) ---
  local a; a="$(avail_mb)"; [ "${a:-99999}" -lt 9000 ] && { flag="${flag}low_mem(${a}MB);"; log "WARN $cell rep$rep: only ${a}MB available"; }
  # --- COLD: the real measurement ---
  python3 "$here/bench-payloads.py" "$req" "$task" "$sections" "$maxtok" >/dev/null
  curl -s --max-time 3000 -w '%{time_total}' -o "$resp" \
    "http://$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$req" \
    > "${base}.wall" 2>/dev/null
  # --- WARM: same prompt, no restart (prefix cache hit), tiny output just to time first token ---
  python3 "$here/bench-payloads.py" "$wreq" "$task" "$sections" 16 >/dev/null
  curl -s --max-time 600 -o "$wresp" \
    "http://$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$wreq" 2>/dev/null
  stop_server
  HERE="$here" RESP="$resp" WRESP="$wresp" CSV="$CSV" WALLF="${base}.wall" FLAG="$flag" \
  RUNID="$RUN_ID" MODEL="$MODEL_TAG" ENGINE="$engine" CELL="$cell" TASK="$task" REP="$rep" python3 - <<'PY'
import json, os, subprocess
resp = os.environ["RESP"]
flags = os.environ.get("FLAG", "")
row = dict(run_id=os.environ["RUNID"], model=os.environ["MODEL"], engine=os.environ["ENGINE"],
           cell=os.environ["CELL"], task=os.environ["TASK"], rep=os.environ["REP"])
try:
    d = json.load(open(resp))
    t = d.get("timings", {}); u = d.get("usage", {})
    row["prompt_tokens"] = u.get("prompt_tokens") or t.get("prompt_n")
    row["completion_tokens"] = u.get("completion_tokens") or t.get("predicted_n")
    row["prefill_tps"] = round(t.get("prompt_per_second", 0) or 0, 2)
    row["decode_tps"] = round(t.get("predicted_per_second", 0) or 0, 2)
    row["ttft_s"] = round((t.get("prompt_ms", 0) or 0) / 1000.0, 2)
except Exception as e:
    for k in ("prompt_tokens", "completion_tokens", "prefill_tps", "decode_tps", "ttft_s"):
        row[k] = ""
    flags += f"parse_error;"
try:  # warm re-send: prefix cache hit -> near-zero prefill / tiny TTFT
    w = json.load(open(os.environ["WRESP"])).get("timings", {})
    row["warm_prefill_tps"] = round(w.get("prompt_per_second", 0) or 0, 2)
    row["warm_ttft_s"] = round((w.get("prompt_ms", 0) or 0) / 1000.0, 2)
except Exception:
    row["warm_prefill_tps"], row["warm_ttft_s"] = "", ""
try:
    row["wall_s"] = round(float(open(os.environ["WALLF"]).read().strip() or 0), 2)
except Exception:
    row["wall_s"] = ""
try:
    v = subprocess.run(["python3", os.path.join(os.environ["HERE"], "bench-verify.py"),
                        os.environ["TASK"], resp], capture_output=True, text=True, timeout=60).stdout.strip()
    p = v.split(" ", 2)
    row["correct"], row["score"] = p[0], (p[1] if len(p) > 1 else "")
except Exception:
    row["correct"], row["score"] = "ERR", ""
# bias detector: a "cold" run on a big prompt can't have a tiny TTFT — that means an accidental cache hit.
try:
    pt = float(row.get("prompt_tokens") or 0); tt = float(row.get("ttft_s") or 0)
    if pt > 4000 and 0 < tt < 1.0:
        flags += "suspect_cold_cache;"
except Exception:
    pass
row["flags"] = flags
cols = ["run_id", "model", "engine", "cell", "task", "rep", "prompt_tokens", "completion_tokens",
        "prefill_tps", "decode_tps", "ttft_s", "warm_prefill_tps", "warm_ttft_s",
        "wall_s", "correct", "score", "flags"]
csv = os.environ["CSV"]
new = not os.path.exists(csv)
with open(csv, "a") as f:
    if new: f.write(",".join(cols) + "\n")
    f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
print("[row] " + "  ".join(f"{c}={row.get(c)}" for c in
      ("engine", "cell", "rep", "prompt_tokens", "completion_tokens", "prefill_tps",
       "decode_tps", "ttft_s", "warm_ttft_s", "wall_s", "correct", "score", "flags")))
PY
}

# ----------------------------------------------------------------------------
log "=== BENCH SUITE $RUN_ID START ===  reps=$REPS out_short=$OUT_SHORT out_long=$OUT_LONG  artifacts=$RDIR"
preflight || { log "=== ABORTED at preflight — box not clean, NOTHING run. Fix and relaunch. ==="; exit 1; }

ENGINES="ik main"
SIZES="600 4k 16k 30k"
declare -A SECT=( [600]=16 [4k]=130 [16k]=520 [30k]=1000 )
QUALITY=1
if [ "${SMOKE:-0}" = "1" ]; then
  ENGINES="ik"; SIZES="600"; REPS=1; QUALITY=0; log "SMOKE MODE: ik / 600 / 1 rep / speed-only"
fi

for eng in $ENGINES; do
  # ---- SPEED grid: input size x output length ----
  for sz in $SIZES; do
    for outl in short long; do
      if [ "$outl" = short ]; then task=needle_short; mt=$OUT_SHORT; else task=needle_long; mt=$OUT_LONG; fi
      for r in $(seq 1 "$REPS"); do run "$eng" "spd_${eng}_${sz}_${outl}" "$task" "${SECT[$sz]}" "$mt" "$r"; done
    done
  done
  # ---- QUALITY grid: 4 tasks ----
  if [ "$QUALITY" = 1 ]; then
    for r in $(seq 1 "$REPS"); do run "$eng" "qual_${eng}_needle" needle_short 520 "$OUT_SHORT" "$r"; done
    for r in $(seq 1 "$REPS"); do run "$eng" "qual_${eng}_summ"   summ        520 1200          "$r"; done
    for r in $(seq 1 "$REPS"); do run "$eng" "qual_${eng}_reason" reason      0   1500          "$r"; done
    for r in $(seq 1 "$REPS"); do run "$eng" "qual_${eng}_code"   code        0   1500          "$r"; done
  fi
done

# ---- FLASH-ATTENTION arm: ik + -fa, long-context cells only (where attention-at-depth dominates) ----
if [ "$QUALITY" = 1 ]; then   # skip in SMOKE
  for sz in 16k 30k; do
    for outl in short long; do
      if [ "$outl" = short ]; then task=needle_short; mt=$OUT_SHORT; else task=needle_long; mt=$OUT_LONG; fi
      for r in $(seq 1 "$REPS"); do run ikfa "spd_ikfa_${sz}_${outl}" "$task" "${SECT[$sz]}" "$mt" "$r"; done
    done
  done
fi

log "Restoring gpt-oss.service"
sudo systemctl start gpt-oss || true
log "=== BENCH SUITE $RUN_ID DONE ===  rows: $CSV"
python3 "$here/bench-report.py" "$CSV" 2>/dev/null || true
