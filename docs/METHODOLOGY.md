# Benchmark suite — design & methodology

Case study: *"how cheaply can you run a sensible LLM on-prem, with receipts."* A reproducible,
fully-documented benchmark proving real numbers AND saving every raw artifact. This doc owns the
benchmark *rationale + methodology*; `scripts/bench-*` own the code; `config.env` owns the values;
`gpt-oss-20b.md` (generated after the run) owns the reader-facing findings; `LAB-NOTEBOOK.md` owns history.

## STATUS / how to run
- **COMPLETE** (2026-06-15): full grid run as `RUN_ID=20260614-191322`, 84/84 runs, no throttling.
  Findings written up in `gpt-oss-20b.md`; data in `results/20260614-191322/`. Re-run instructions below.
- Harness built & validated 2026-06-14, synced to CT.
- Smoke first: `SMOKE=1 bash ~/gpt-oss/scripts/bench-suite.sh` (ik / 600 / 1 rep / speed-only, ~5 min).
- Full grid (detached, ~6–9 h): `mkdir -p ~/gpt-oss/results && nohup bash ~/gpt-oss/scripts/bench-suite.sh >> ~/gpt-oss/results/suite.out 2>&1 &`
- PAUSE: `pkill -f bench-suite.sh ; pkill -f 'llama-server.*--port 8080'` (this leaves gpt-oss.service
  stopped — run `sudo systemctl start gpt-oss` if you want to serve in the meantime).
- RESUME: `RUN_ID=<id> nohup bash …/bench-suite.sh >> ~/gpt-oss/results/suite.out 2>&1 &` — skips every
  cell-rep already present in results.csv (the row is the authoritative "done" marker, written last;
  an interrupted run is re-done cleanly, never half-counted). Finished work is never repeated.
- NO CT reboot anywhere: a fresh llama-server PROCESS per run gives cold KV + cold prompt cache; the
  model stays in page cache (fast, consistent load). The suite stops gpt-oss.service for the duration
  and restarts it at the end (a manual pause leaves it stopped — restart it yourself or just resume).
- After completion: `python3 ~/gpt-oss/scripts/bench-report.py <results.csv>` → tables → author `gpt-oss-20b.md`.

## Harness (in scripts/; values from config.env SSOT)
- `bench-payloads.py` — SINGLE OWNER of all prompts + fixtures (needle 4417, reason ans 53, 8 summ
  figures). Deterministic; importable. `bench-payloads.py <out> <task> [sections] [max_tokens]`.
- `bench-verify.py <task> <resp.json>` — task correctness → `PASS|FAIL <score> <note>`. Code task
  execs the model's `is_balanced` against 8 cases in a restricted sandbox (denylist on import/os/…).
- `bench-suite.sh` — orchestrator, runs ON the CT. Per run: fresh server (cold cache) → warm-up
  (different prompt) → curl localhost → save raw req/resp → verify → append results.csv → kill.
  Stops gpt-oss.service for the duration, restores at end. Idempotent/resumable by RUN_ID.
- `bench-report.py <results.csv>` — median markdown tables (data for gpt-oss-20b.md).

## Tests & metrics — what each prompt is and WHY it looks that way
Two prompt FAMILIES; `bench-payloads.py` is their single owner (deterministic — same args → identical bytes).

SPEED family (task `needle_short` / `needle_long`) — purpose: isolate the two perf axes.
- Structure: a filler "document" of N numbered sections (~30.5 tok/section: N=16/130/520/1000 →
  ~600/4k/16k/30k prompt tokens) + a question. Input size is the *only* thing that varies across
  size cells, so prefill cost maps cleanly to prompt length.
- Why a needle in the filler (not just raw filler): it makes every speed run *also* a correctness
  check — the model must retrieve a planted fact, so a fast-but-broken run can't pass silently.
- `needle_short`: asks "reply with ONLY the number" → ~tens of output tokens → measures the
  PREFILL axis (decode is negligible). `needle_long`: "state the number, THEN write a 1000-word
  essay on how CPUs execute instructions" → forces sustained generation up to `OUT_LONG` (2000)
  → measures the DECODE axis at that input depth. Same doc, two output regimes = clean 2D grid.
- Why jersey number 4417 (not a "code"/"secret", not an odd name): "authorization code/secret"
  framing trips GPT-OSS safety refusals; odd proper names get normalized by the tokenizer/model.
  A benign jersey number is exactly reproducible and unambiguous to grade. System prompt states the
  doc is fictional and shareable to pre-empt refusal. (Learned the hard way — see WORKLOG 18:27.)

QUALITY family — purpose: prove the model is actually USEFUL, not just fast, with auto-gradable tasks.
- T1 `needle` (retrieval @ ~16k): the 520-section needle prompt, short answer. Tests long-context
  recall in isolation. Grade: exact substring `4417` present. (Mirrors the 16k speed cell but listed
  separately so the quality table reads as a capability matrix.)
- T2 `summ` (summarization @ ~16k): a quarterly-report doc carrying 8 distinct sentinel FIGURES
  (per-division sales, exports, headcount, downtime, satisfaction) padded with filler. Asks for a
  ~200–300w exec summary "including every specific figure." Grade: recall = how many of the 8
  figures survive into the summary; PASS ≥ 0.75. Why figures: numeric recall is unambiguous to score,
  unlike prose-quality judgement.
- T3 `reason` (math/reasoning): a small multi-step word problem (answer 53) requiring "end with
  `ANSWER: <n>`". Why the fixed format: lets the grader extract the final answer deterministically
  and ignore the reasoning-channel chatter. Grade: last `ANSWER:` integer == 53.
- T4 `code` (coding): "write `is_balanced(s)` for ()[]{}". Why this task: it has a crisp correct/
  incorrect boundary (naive count-the-brackets fails nesting like `([)]`). Grade: the model's code is
  EXECUTED against 8 cases in a restricted sandbox (denylist on import/os/eval/open/… → unsafe code
  is rejected, not run); PASS = 8/8. This is real functional correctness, not pattern-matching.
- All fixtures (4417, answer 53, the 8 figures, the 8 code cases) live ONCE in bench-payloads.py;
  bench-verify.py imports them so prompt and grader can never drift apart.

What we MEASURE (per run, saved raw + in results.csv) — all from llama-server's own `timings`/`usage`:
- `prompt_tokens` / `completion_tokens` — input depth actually seen / output actually produced
  (completion counts BOTH the reasoning channel and the visible answer — GPT-OSS spends real tokens
  thinking, so this is the honest decode workload).
- `prefill_tps` (= timings.prompt_per_second) — prompt-processing throughput; the prefill axis.
- `decode_tps` (= timings.predicted_per_second) — token-generation throughput; the decode axis.
- `ttft_s` (≈ timings.prompt_ms/1000) — COLD time-to-first-token = how long before output starts; this
  is the latency a *user* feels on a fresh long prompt, prefill-dominated (the real cost at depth).
- `warm_prefill_tps` / `warm_ttft_s` — same prompt re-sent without restart (prefix cache HIT) = the
  realistic follow-up latency once the big context is cached. Cold-vs-warm is the headline of this study.
- `wall_s` (curl time_total) — end-to-end request wall-clock; the number you actually wait.
- `flags` — anomaly markers (multi_instance / low_mem / suspect_cold_cache / parse_error); empty = trustworthy.
- `correct` / `score` — task verdict from bench-verify.py (proves the speed wasn't bought with garbage).
Reps are summarized by MEDIAN (robust to a single slow outlier from a background blip).

## Fixed rig (document verbatim in gpt-oss-20b.md)
- HW: Dell T5810, Xeon E5-2680v4 (14c/28t Broadwell, AVX2, NO AVX-512/AMX), 64 GB DDR4-2400
  quad-channel (4×16 GB), CPU-only, no GPU. (~2016-era; cite used-market cost in RESULTS.)
- CT: unprivileged LXC (CTID `$CTID`), Debian 13, cpuset pinned 0-11 (12 physical cores), 24 GB cap, NVMe.
- Serve config (config.env SSOT): ctx `$CTX`(32768), `--parallel $SLOTS`(1), `--threads $THREADS`(11),
  `$FA`(off), sampling `$TEMP/$TOP_P/$TOP_K/$MIN_P`, `--jinja`. Model `$MODEL_FILE` (11.27 GiB).
- Engines: mainline llama.cpp (`$LLAMA_COMMIT` 6e14286ed) vs ik_llama.cpp (`$IK_COMMIT` 670a3f6).

## The test matrix — what gets run, and where "84 runs" comes from
A **cell** is one exact combination we measure — e.g. *"ik engine, ~16k-token input, short output"*.
Every cell is run **3 times** ("3 reps") and the three are summarized by their **median** (so one
background hiccup can't skew a result). So: **runs = cells × 3**. The cells fall into three groups:

| group | what is varied | cells | runs (×3) |
|---|---|---|---|
| **Speed** | 2 engines (mainline, ik) × 4 input sizes (~600, ~4k, ~16k, ~30k tokens) × 2 output lengths (short ≈64 tok, long ≈2000 tok) | 16 | 48 |
| **Quality** | 2 engines × 4 tasks (needle, summarize, reason, code) | 8 | 24 |
| **Flash-attn** | ik + `-fa` only, at the 2 deep inputs (~16k, ~30k) × 2 output lengths | 4 | 12 |
| | | **28** | **84** |

- **Speed** isolates the two performance axes: input size drives *prefill*, output length drives *decode*.
- **Quality** proves the model is actually useful, not just fast (each task is auto-graded).
- **Flash-attn** is tested *only* on the deep inputs and *only* on ik, because `-fa` can only help where
  long-context attention dominates; crossing it with the entire grid would add ~72 pointless runs
  (settled with the user — see WORKLOG 2026-06-14).

Each of the 84 runs also does the per-run restart + throwaway warm-up + cheap warm re-send (above).
Total wall time ≈ **9–11 h** (one overnight); the deepest cold prefill alone is ~13 min (ik) / ~25 min
(mainline). User accepts the compute/electricity cost.

## Cold vs WARM (TTFT) — why and how
TTFT on a long prompt is prefill-dominated, so a cold full-32k prompt is genuinely minutes to first
token — but that's a once-per-fresh-context cost. In real chat/RAG the big prefix is cached and
follow-ups skip it. To show BOTH numbers with receipts: each run measures COLD (fresh server), then
immediately RE-SENDS the identical prompt WITHOUT restarting (`max_tokens=16`) → prefix cache HIT →
records `warm_prefill_tps` / `warm_ttft_s`. The warm re-send is near-free (prefill ≈ 0 + 16 tokens),
so it adds ~10–15 min across the whole grid, not a second pass. (Note: the per-run warm-up uses a
DIFFERENT prompt on purpose, to keep the cold measurement cold; warm is this separate same-prompt send.)

## What happens in one measured run (one engine × one cell × one rep)
Read this as **three setup steps that record nothing**, then **two measurements that do**, then save.
The numbers in results.csv come ONLY from steps 4 and 5 — never from the warm-up.

**Setup (nothing recorded):**
1. **Clean slate.** Kill any running server; launch this cell's engine from scratch and wait until it
   answers `/health`. A brand-new process has empty caches, so no earlier test can leak into this one.
   Then assert exactly one server is running (more than one = contention = biased numbers).
2. **Throwaway warm-up — not a measurement.** Send one tiny, *unrelated* prompt (literally
   "reply: ready") and discard the answer. This absorbs one-time startup cost (model paging into RAM,
   threads spinning up) so it can't distort the real run. Because it's a *different* prompt, it does
   **not** put the test prompt into the cache — the test prompt stays "cold".
3. **Headroom check.** Confirm enough free RAM (flag the row if not).

**Measurements (recorded):**
4. **COLD send — the main measurement.** Send the real test prompt for the *first* time and record it.
   This is the honest "first time you ask this" result (caches cold). The request goes to
   `127.0.0.1:$PORT` *on the CT itself*, so no network/SSH latency contaminates the timing
   (`curl --max-time 3000`). → records cold prefill t/s, decode t/s, **cold TTFT**, wall-clock.
5. **WARM send — the same prompt again.** Immediately re-send the *identical* prompt, no restart, asking
   for only 16 tokens. The server still has this prompt cached, so it skips re-reading it → near-instant.
   This is the realistic "ask again / context already cached" result. → records warm prefill, **warm TTFT**.

**Save:**
6. Check correctness (task-specific grader), append one row to results.csv, keep the raw request+response
   JSON for *both* sends (full content + reasoning channel + usage + timings), then kill the server.

The point of steps 4 vs 5: the same prompt gives a **cold** number (first ask, prefill-dominated, slow at
depth) and a **warm** number (cached, near-instant). That cold-vs-warm contrast is the study's headline.

## Resilience / self-defense (why no reboots are needed)
An unattended ~9 h run must not silently produce empty or biased data. Reboots don't prevent the
real failure modes (orphan processes, hangs) — a fresh PROCESS per run is the real isolation, and the
harness defends itself instead:
- PREFLIGHT clean-slate gate (once, + on resume): aborts the whole suite if a stray server is running,
  the port is busy, or <15 GB RAM is available — refuses to burn the night on a dirty box.
- stop_server escalates SIGTERM→SIGKILL and verifies the process is gone + port free, before AND after
  every run (a wedged server can't bleed into the next cell).
- Single-instance assertion before each measurement: if ever ≠1 server (the contention-bias culprit),
  it cleans + retries once, and FLAGS the row if still wrong.
- Memory-headroom check per run; cold-cache bias detector (a "cold" run on a >4k prompt with sub-1s
  TTFT = accidental cache hit → flagged).
- Every anomaly lands in a `flags` CSV column; the end-of-run **Data health** summary (from
  bench-report.py) lists rows-captured-vs-expected, PASS count, and every flagged/non-pass row — so the
  night is trustworthy at a glance. Resume keys on the CSV row, so a kill (or even a reboot) loses nothing.

## Pitfalls already learned (baked into harness)
- Cache: never re-send identical prompt without restart → cache hit = bogus prefill (saw 16 vs 39 t/s).
- Refusals: benign + exactly-reproducible needles only (jersey number 4417; no "secret/code"; no odd names).
- Long output: GPT-OSS spends tokens in reasoning channel — count+save them; raise max_tokens (~2000).
- nohup the harness; verify single server instance (pgrep self-matches inflate counts).
- 11 threads keeps a core for sshd; run harness ON the CT writing artifacts to files, collect at end.

## Artifacts & deliverable
- `gpt-oss/results/<run-id>/` (DATA): per-run raw cold+warm req/resp JSON
  (`raw/<cell>_rep<r>.{req,resp}.json` + `.warm.{req,resp}.json`), `results.csv`, `server.log`, `suite.out`.
- `gpt-oss/gpt-oss-20b.md`: the case study — rig, matrix, tables, cost framing, and links to raw proofs.
- Harness scripts in `scripts/` become part of the blueprint.

## Decisions made (were open at design time)
- Server lifecycle: harness launches the engine binary DIRECTLY per run (not via systemd) → cleanest
  isolation + cold cache every run; systemd gpt-oss.service is stopped for the suite, restored at end.
- Engine switch: `bin_for()` maps `ik`/`main` → the two build paths (`$IK_DIR`/`$LLAMA_DIR`); no
  config.env edit per cell.
- Cold cache between REPS too (not just cells): same prompt re-sent without restart = cache hit, so a
  fresh process is launched for every single measured run (the 3 reps included).
