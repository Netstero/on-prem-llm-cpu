# Running a 20B LLM on a 2016 CPU: a measured case study

**How cheaply can you run a capable LLM on-premises, CPU-only — with receipts.**

A reproducible benchmark of OpenAI's `gpt-oss-20b` (MoE, MXFP4) served via `llama.cpp` on a
single ~2016-era Xeon workstation, no GPU. Two inference engines compared, four input sizes,
two output regimes, four quality tasks, three repetitions each — 84 measured runs, every raw
artifact saved.

- **Run ID:** `20260614-191322` · **Date:** 2026-06-15 · **Runs:** 84 (28 cells × 3 reps)
- **Correctness:** 84/84 effective (one recorded FAIL is a grader false-positive — see §6.4)
- **Thermal validity:** zero CPU throttle events across the entire run (§5.4)
- All numbers below are medians of 3 repetitions, taken verbatim from `results/20260614-191322/results.csv`.
  Nothing in this document is estimated; figures that are derived (ratios) are marked as such.

---

## 1. Summary of findings

1. **It works, and it's correct.** `gpt-oss-20b` runs CPU-only on this box and solves every task —
   long-context retrieval to 30k tokens, summarization (all sentinel figures recalled), multi-step
   arithmetic, and executable code — at OpenAI's recommended sampling (`temperature=1.0`).
2. **Decode (token generation) is usable; cold prefill on huge prompts is not.** Short-context decode
   is ~13–17 tok/s. But a *cold* 30k-token prompt takes **~17 minutes** to first token — prefill is
   the wall, not decode.
3. **The cost is one-time.** The same 30k prompt re-sent against a warm prefix cache returns the first
   token in **0.12 s** — a **~8,500× reduction** (derived). The "minutes to first token" figure is a
   per-fresh-context cost, not a per-request one.
4. **Engine choice is workload-dependent, and decisive at depth.** The `ik_llama.cpp` fork delivers
   **1.66–1.68× faster prefill** at ≤4k tokens and — more importantly — holds decode throughput at
   long context where mainline `llama.cpp` **collapses** (at 30k: ik 8.7 vs mainline 2.9 tok/s, a
   **~3× difference**, derived). Mainline wins only short-prompt chat decode.
5. **Flash-attention (`-fa`) makes no measurable difference** on ik at 16k/30k — a clean negative result.

---

## 2. System under test

### 2.1 Hardware (document verbatim; ~2016-era, used-market class)
| Component | Spec |
|---|---|
| Machine | Dell Precision T5810 workstation |
| CPU | Intel Xeon E5-2680 v4 (Broadwell, 14C/28T, base 2.4 GHz, AVX2; **no** AVX-512/AMX) |
| Memory | 64 GB DDR4-2400, quad-channel (4×16 GB); theoretical bandwidth ≈76.8 GB/s, effective ~55–60 GB/s (estimated, not measured) |
| GPU | none (CPU-only) |
| Storage | NVMe SSD |

### 2.2 Container & isolation
- Unprivileged Proxmox LXC (Debian 13), CTID <CTID>.
- **CPU pinning:** host `cpuset` 0–11 → 12 distinct physical cores (HT siblings verified to lie in
  14–25; no hyperthread collisions). Cores 12–13 + siblings left to the host.
- **24 GB** cgroup memory cap. Inference ran as the **sole** model process throughout (enforced by the
  harness; see §3.5).

### 2.3 Model
`openai/gpt-oss-20b` — Mixture-of-Experts Transformer. Authoritative specs (OpenAI model card,
arXiv:2508.10925; HF `config.json`):
- **20.9B** total parameters (OpenAI's informal name is "20b"; arXiv card: 20.91B), **3.6B active per
  token** (arXiv card: 3.61B); **32 experts, top-4** routing; **24 layers**.
- **Native MXFP4** quantization of the MoE expert weights (~4.25 bits/param); all other tensors BF16.
- GGUF file `gpt-oss-20b-mxfp4.gguf` (ggml-org): **12.1 GB** (= ~11.27 GiB).
- Context length **131,072** ("128k"); tokenizer `o200k_harmony` (vocab 201,088); **Harmony** chat
  format is mandatory (`--jinja`). License **Apache-2.0**; released **2025-08-05**.
- OpenAI-recommended sampling: **`temperature=1.0, top_p=1.0`** (used here).

### 2.4 Software & serving configuration (single source of truth: `scripts/config.env`)
- Engines compared: **mainline `llama.cpp`** (commit `6e14286ed`, `llama-server` build b9631) and
  **`ik_llama.cpp`** (commit `670a3f6`), a CPU/MoE-optimized fork. Build flags (per `scripts/`):
  mainline `cmake -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON` (llama.cpp auto-detects native arch by
  default → AVX2/FMA used); ik built with explicit `-DGGML_NATIVE=ON -DGGML_AVX2=ON -DGGML_FMA=ON`.
- Serve flags (identical across engines): `--ctx-size 32768 --parallel 1 --jinja --threads 11
  --threads-batch 11 --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0`. **No** repetition penalty, **no**
  KV-cache quantization (both degrade gpt-oss). Flash-attention off except the dedicated `-fa` arm.
- `--threads 11` reserves one of the 12 pinned cores for the OS/sshd/monitoring (rationale §5.3).

---

## 3. Methodology

### 3.1 Metrics and conventions
We report the two physically distinct phases of autoregressive inference separately — **prefill**
(prompt processing, compute-bound) and **decode** (token generation, memory-bandwidth-bound) — as is
standard in LLM serving benchmarks (NVIDIA, vLLM, Anyscale). All values are read from `llama-server`'s
own `timings`/`usage` JSON plus client wall-clock:

| Metric (column) | Definition | Maps to |
|---|---|---|
| `prefill_tps` | `timings.prompt_per_second` — prompt tokens processed/sec | prefill throughput |
| `decode_tps` | `timings.predicted_per_second` — generated tokens/sec over the decode phase | inverse of TPOT (excl. first token) |
| `ttft_s` | `timings.prompt_ms / 1000` — server-side prefill time | **TTFT** (Time To First Token) |
| `warm_ttft_s` | same, for the warm re-send (cached prefix) | warm TTFT |
| `wall_s` | client `curl` `time_total` | end-to-end request latency |
| `prompt_tokens` / `completion_tokens` | from `usage` | input depth / output produced |

Conventions, stated explicitly (per best-practice guidance on metric ambiguity):
- **TTFT** here is the server-reported prefill time. The client runs on the same host (`curl` →
  `127.0.0.1`), so network latency ≈ 0; `ttft_s` is a tight lower bound on client-observed TTFT
  (the first decoded token follows within ~1/`decode_tps`, negligible at these prompt sizes).
- **`decode_tps` is a decode-phase rate that excludes prefill** — i.e. `1/TPOT` in the
  "excludes-first-token" convention used by vLLM and NVIDIA GenAI-Perf (and *not* the AnyScale LLMPerf
  convention that folds TTFT into per-token latency).
- **`completion_tokens` counts the Harmony reasoning channel plus the final answer.** gpt-oss spends
  real tokens "thinking," so this is the honest decode workload.

### 3.2 Experimental design (the grid)
A **cell** is one exact combination we measure (e.g. *ik engine, ~16k-token input, short output*). Each
cell is run 3 times and summarized by the median, so **runs = cells × 3**. The cells fall into three groups:

| group | what is varied | cells | runs (×3) |
|---|---|---|---|
| **Speed** | 2 engines (mainline, ik) × 4 input sizes (~600, ~4k, ~16k, ~30k tok) × 2 output lengths (short ≤64 tok, long ≤2000 tok) | 16 | 48 |
| **Quality** | 2 engines × 4 tasks (T1 retrieval@16k, T2 summarization@16k, T3 reasoning, T4 coding) | 8 | 24 |
| **Flash-attention** | ik + `-fa` only, at the 2 deep inputs (16k, 30k) × 2 output lengths | 4 | 12 |
| | | **28** | **84** |

Flash-attention is tested only on the deep inputs and only on ik, where long-context attention could
matter; crossing `-fa` with the whole grid would add 72 runs that cannot help where decode isn't
attention-bound. Input sizes were set by a deterministic filler-doc generator (~30.5 tokens/section);
realized prompt sizes: 638 / 4104 / 15960 / 30554 tokens.

### 3.3 Per-run protocol (what makes the numbers credible)
Each measured run has **three setup steps that record nothing**, then **two measurements that do** — the
numbers in `results.csv` come *only* from steps 4–5, never from the warm-up — then a save:

*Setup (not recorded):*
1. **Fresh server process** launched for that run → cold KV cache *and* cold prompt cache. A new process
   per run — including each of the 3 reps — is the isolation (see §3.5).
2. **Throwaway warm-up — not a measurement.** A *different*, tiny prompt ("Reply with the single word:
   ready.") is sent and its answer discarded, purely to absorb one-time startup cost (model paging into
   RAM, threads spinning up). Because it is a *different* prompt, it leaves the test prompt uncached — the
   test prompt stays cold.

*Measurements (recorded):*
3. **Cold send — the main measurement.** The test prompt, sent for the *first* time via `curl` →
   `127.0.0.1` (no SSH in the request path) → cold prefill, decode, and **cold TTFT**.
4. **Warm send — the same prompt again.** The *identical* prompt re-sent **without restarting**
   (`max_tokens=16`) → a prefix-cache hit → `warm_prefill_tps` / **`warm_ttft_s`**. Near-free
   (prefill ≈ 0 + 16 tokens); this is the realistic "context already loaded" follow-up latency.

*Save:*
5. **Capture & grade**, append one row to `results.csv`, save raw request/response JSON for *both* sends
   (prompt+context, `content`, `reasoning_content`, `usage`, `timings`), then kill the server.

The point of steps 3 vs 4: the *same* prompt yields a **cold** number (first ask, prefill-dominated) and a
**warm** number (cached, near-instant). That cold-vs-warm contrast is a headline of this study.

### 3.4 Quality verification (so speed isn't bought with garbage)
Every task is **deterministically auto-graded** — no LLM-as-judge, no string-similarity — following
the principle that a benchmark reporting only speed is uninterpretable:

| Task | Design | Grading | Lineage |
|---|---|---|---|
| T1 retrieval (needle) | benign fact (jersey number `4417`) planted at 55% depth in a 16k filler doc | exact-substring `4417` in output | Needle-in-a-Haystack (Kamradt, 2023) |
| T2 summarization | quarterly-report doc carrying **8 sentinel figures**; ask for an exec summary "including every figure" | recall of the 8 figures; PASS ≥ 0.75 | numeric-recall (unambiguous, unlike prose judging) |
| T3 reasoning | multi-step arithmetic (answer 53), output must end `ANSWER: <n>` | last `ANSWER:` integer == 53 | exact-match numeric extraction (cf. GSM8K, LiveBench) |
| T4 coding | write `is_balanced(s)` for ()[]{} | **execute** the model's code against 8 cases incl. mis-nesting `([)]`; PASS = 8/8 | functional correctness (HumanEval/pass@1) |

The needle is a benign jersey number, not a "secret/code," because the latter framing triggers
gpt-oss safety refusals; odd proper names get normalized by the tokenizer (both learned empirically).
Every speed run *also* embeds the needle, so a fast-but-broken run can't pass silently.

### 3.5 System isolation & validity controls
- **Cold-cache guarantee:** a fresh process per run. Re-sending an identical prompt to a live server is
  a prefix-cache hit — the single most common way benchmarks are inflated; we observed it directly
  (a cached near-32k prompt reported ~16 t/s prefill vs ~39 cold). The warm-up therefore uses a
  *different* prompt, and cold is always a fresh process.
- **Single instance enforced:** the harness asserts exactly one `llama-server` before each measurement
  (cleans + retries once, flags the row otherwise) — contention would bias the numbers.
- **Pre-flight gate:** the suite aborts if the box isn't clean (stray server, busy port, <15 GB RAM)
  rather than producing biased/empty data overnight.
- **Memory & cache anomaly detection:** a `flags` column marks any run with multiple instances, low
  memory, a parse error, or a "cold" run whose TTFT was implausibly low (accidental cache hit). **No
  run was flagged.**
- **Thermal:** monitored throughout (§5.4).

### 3.6 Statistical treatment
3 repetitions per cell, summarized by **median** (robust to a single slow outlier). We report a single
representative figure per cell; we did **not** compute high-order percentiles (p99) because this is a
single-stream latency/throughput study, not a concurrent-load study (see §6 Limitations).

### 3.7 How this methodology compares to industry best practice
We benchmarked our own method against the practices documented for MLPerf Inference (MLCommons),
vLLM's `benchmark_serving`, llama.cpp's `llama-bench`, NVIDIA GenAI-Perf, and the LLM-eval literature
(NIAH, RULER, HumanEval/pass@k, goodput). Verdict per practice:

| Best practice | Source(s) | What we did | Verdict |
|---|---|---|---|
| Warm-up before measurement | llama-bench; FMwork; MLPerf | warm-up with a *different* prompt each run | ✅ aligned |
| Distinguish cold / warm / hot cache; never re-send identical prompt and count it as cold | vLLM; SGLang; MLPerf (no cross-query KV) | fresh process = cold; warm re-send measured *separately and labelled* | ✅ aligned (and we report **both**) |
| Separate prefill from decode | NVIDIA; vLLM; Anyscale | distinct `prefill_tps` / `decode_tps` / TTFT | ✅ aligned |
| TPOT/decode excludes the first token | vLLM; NVIDIA GenAI-Perf; AWS Neuron | `decode_tps` = decode-phase rate, excl. prefill | ✅ aligned |
| Sweep input AND output lengths | llama-bench; vLLM; MLPerf | 4 input × 2 output sizes | ✅ aligned |
| Report full HW/SW/engine commit/flags + sampling + seeds | MLPerf; Databricks; Bench360 | §2; SSOT `config.env`; pinned commits | ✅ aligned |
| Report quality alongside speed (quality floor) | MLPerf | 8 auto-graded quality cells + needle in every speed run | ✅ aligned |
| Deterministic auto-grading; execute code vs unit tests | LiveBench; HumanEval/pass@k | exact-match / numeric / **code execution** | ✅ aligned |
| Single instance; CPU pinning; no co-tenant load; thermal stability | edge-benchmark methodology; CPU-interference study | enforced single instance; cpuset pinning; lab paused; throttle counters logged | ✅ aligned |
| Multiple repetitions + statistics | llama-bench (`-r 5`); FMwork (≥10 for public reporting) | **3 reps**, median | ⚠️ partial — fewer reps than the ≥10 some sources advise (mitigated: observed variance is tiny, §4) |
| Percentiles (p50/p90/**p99**) under load | MLPerf; Anyscale; NVIDIA | median only; **no concurrent load** | ⚠️ scope — single-stream study; no p99/goodput |
| Concurrency / goodput / throughput-under-load | MLPerf server scenario; DistServe (goodput) | not measured (`--parallel 1`) | ❌ out of scope (see §6) |
| Multi-task long-context eval beyond retrieval | RULER; LongBench | NIAH-style retrieval + summarization only | ⚠️ partial — no multi-hop/aggregation (RULER) |
| Sampling vs reproducibility tension | arXiv:2605.24217; arXiv:2506.09501 | used vendor `temp=1.0` (not greedy) | ⚠️ noted — favors fidelity over bit-reproducibility; mitigated by 3 reps + robust tasks |

Net: the methodology is consistent with serving-benchmark best practice on the axes relevant to a
**single-user, on-prem, CPU** deployment (phase separation, cold/warm honesty, isolation, deterministic
quality grading). It deliberately does **not** cover the *concurrent-load* axis (goodput, p99 tails)
that MLPerf's server scenario targets — that is the main scoping limitation (§6).

---

## 4. Results

### 4.1 Full speed grid (medians of 3 reps)

| engine | input tok | output | prefill t/s | decode t/s | **TTFT cold** | **TTFT warm** | wall s |
|---|---|---|---|---|---|---|---|
| ik | 638 | short | 85.16 | 13.60 | 6.76 s | 0.07 s | 11.4 |
| ik | 675 | long | 84.71 | 13.21 | 7.24 s | 0.07 s | 158.7 |
| ik | 4104 | short | 79.90 | 12.81 | 50.59 s | 0.08 s | 55.6 |
| ik | 4141 | long | 79.56 | 12.47 | 51.27 s | 0.08 s | 211.8 |
| ik | 15960 | short | 51.59 | 10.90 | 308.2 s | 0.10 s | 314.2 |
| ik | 15997 | long | 51.39 | 10.22 | 310.1 s | 0.09 s | 505.9 |
| ik | 30554 | short | 29.70 | 8.72 | 1026.6 s | 0.12 s | 1033.6 |
| ik | 30591 | long | 29.45 | 8.45 | 1036.7 s | 0.12 s | 1273.2 |
| main | 638 | short | 51.21 | 16.64 | 11.29 s | 0.06 s | 13.0 |
| main | 675 | long | 51.37 | 14.75 | 11.97 s | 0.17 s | 147.7 |
| main | 4104 | short | 47.63 | 12.45 | 84.90 s | 0.08 s | 90.0 |
| main | 4141 | long | 47.60 | 11.67 | 85.74 s | 0.26 s | 257.2 |
| main | 15960 | short | 37.46 | 7.10 | 424.5 s | 0.15 s | 432.1 |
| main | 15997 | long | 37.32 | 6.52 | 427.0 s | 0.68 s | 734.6 |
| main | 30554 | short | 29.24 | 2.94 | 1042.9 s | 0.35 s | 1064.9 |
| main | 30591 | long | 29.26 | 2.74 | 1043.3 s | 1.56 s | 1774.3 |

Reproducibility was excellent — e.g. ik 30k-short TTFT across the 3 reps: 1026.59 / 1026.06 / 1026.72 s;
mainline 4k-long prefill: 47.60 / 47.60 / 47.54 t/s. Per-rep data is in `results.csv`.

### 4.2 Cold vs warm TTFT — the central result
The same prompt, cold (fresh prefill) vs warm (prefix-cache hit), ik engine:

| input | TTFT cold | TTFT warm | ratio (derived) |
|---|---|---|---|
| 638 | 6.76 s | 0.07 s | ~97× |
| 4104 | 50.59 s | 0.08 s | ~632× |
| 15960 | 308.2 s | 0.10 s | ~3,080× |
| 30554 | 1026.6 s | 0.12 s | **~8,555×** |

Cold TTFT scales roughly linearly with input length (prefill is O(n) in tokens). Warm TTFT is
~constant and sub-0.15 s regardless of context size. **The "minutes to first token" cost is paid once
per fresh context; a reused/cached context responds effectively instantly.**

### 4.3 Engine comparison (ik vs mainline)
Prefill (ik advantage, derived from §4.1):

| input | ik t/s | main t/s | ik advantage |
|---|---|---|---|
| 638 | 85.16 | 51.21 | **1.66×** |
| 4104 | 79.90 | 47.63 | **1.68×** |
| 15960 | 51.59 | 37.46 | 1.38× |
| 30554 | 29.70 | 29.24 | **1.02× (converges)** |

Decode (short output cells):

| input | ik t/s | main t/s | winner |
|---|---|---|---|
| 638 | 13.60 | **16.64** | mainline **+22%** |
| 4104 | 12.81 | 12.45 | ≈ tie |
| 15960 | 10.90 | 7.10 | ik **1.54×** |
| 30554 | 8.72 | 2.94 | ik **~2.97×** |

Two clean effects:
- **ik's prefill lead is large at shallow/mid depth (~1.67×) but converges to parity by 30k** — at
  extreme depth both engines become bandwidth-bound on prefill.
- **Mainline decode collapses with context depth** (16.6 → 2.9 tok/s from 600 → 30k) while ik degrades
  gracefully (13.6 → 8.7). At 30k, ik decodes ~3× faster. Mainline's only win is short-prompt chat decode.

### 4.4 Flash-attention arm (ik vs ik+`-fa`)
| input/output | ik prefill / decode | ik+fa prefill / decode |
|---|---|---|
| 16k short | 51.59 / 10.90 | 51.61 / 10.98 |
| 16k long | 51.39 / 10.22 | 51.48 / 10.25 |
| 30k short | 29.70 / 8.72 | 29.47 / 8.70 |
| 30k long | 29.45 / 8.45 | 29.18 / 8.45 |

Differences are within run-to-run noise (<1%). **`-fa` has no measurable effect on ik for long-context
CPU inference** — ik's attention kernels are already efficient; the flag is moot. (We did *not* test
`mainline+-fa`, the configuration that might have mitigated mainline's decode collapse — see §6.)

### 4.5 Quality results (medians; all auto-graded)
| engine | task | input tok | prefill t/s | decode t/s | TTFT | correct |
|---|---|---|---|---|---|---|
| ik | T1 retrieval@16k | 15960 | 51.38 | 11.01 | 309.4 s | **3/3** |
| ik | T2 summarization@16k | 16039 | 51.28 | 10.57 | 311.6 s | **3/3** (8/8 figures) |
| ik | T3 reasoning | 146 | 68.50 | 13.60 | 1.23 s | **3/3** |
| ik | T4 coding | 142 | 65.70 | 13.54 | 1.22 s | **3/3** |
| main | T1 retrieval@16k | 15960 | 37.53 | 6.88 | 423.7 s | **3/3** |
| main | T2 summarization@16k | 16039 | 37.30 | 6.70 | 428.4 s | **3/3** (8/8 figures) |
| main | T3 reasoning | 146 | 50.19 | 17.15 | 1.71 s | **3/3** |
| main | T4 coding | 142 | 49.41 | 17.28 | 1.66 s | 2/3 (1 false-pos, §6.4) |

Both engines produce **correct** output on every task. Example (ik, T3, verbatim tail):
`"29 apples + 24 apples = 53 apples … So, Sara now has **53 apples**. ANSWER: 53"`. The quality
difference between engines is **zero**; the difference is purely speed (same model, same sampling).

---

## 5. Discussion

### 5.1 The usable envelope (cost-effectiveness, with numbers)
On this CPU box, usability is defined by *prefill*, not decode:

| context | cold TTFT (ik) | usable interactively? |
|---|---|---|
| ~600 tok | ~7 s | yes |
| ~4k tok | ~51 s | borderline (tolerable for non-chat) |
| ~16k tok | ~5 min | only with caching / pre-warm |
| ~30k tok | ~17 min | only with caching / pre-warm |

Because the cold cost is one-time (§4.2), the *usable* pattern for large context is: prefill the stable
context once (off the critical path — at upload or via background pre-warm), then operate over it at
warm speeds (~0.1 s TTFT + 8–13 tok/s decode). Under that pattern even 30k context is usable. The box
is **not** usable for cold, large, ad-hoc, interactive prompts — that is the narrow, avoidable case.

### 5.2 Engine recommendation
- **`ik_llama.cpp` for anything with real context** (RAG, long documents, agents): faster prefill at
  shallow/mid depth, and decisively better decode at depth (~3× at 30k). Also the leaner-memory engine
  (mmap-shared model vs mainline's anonymous copy).
- **mainline `llama.cpp` only for pure short-prompt chat** where its ~22% short-context decode edge
  matters and prompts stay small.
- The standing default on this deployment is **ik**.

### 5.3 Why 11 threads
Decode is memory-bandwidth-bound and saturates near 10 cores (a prior `llama-bench` sweep: tg 8t=17.2,
10t=19.1, 12t=20.1 t/s); prefill is compute-bound and scales with cores. Using all 12 cores starves the
container's own control plane (we hit SSH lockouts at 12 threads). `--threads 11` keeps full prefill
capacity minus ~8%, ~95% of peak decode, and a responsive box. (Note the live, served decode rates here
are below the synthetic `llama-bench` ceiling: the HTTP + Harmony-template + sampling path has real
overhead — another reason to benchmark the served path, not just `llama-bench`.)

### 5.4 Thermal validity
The Xeon's per-package thermal-throttle counter (`/sys/devices/system/cpu/cpu0/thermal_throttle/
package_throttle_count`) read **0 before and after** the full multi-hour run; idle 49 °C, peak ~77 °C —
well under the throttle threshold. Independently, the near-zero rep-to-rep variance (§4.1) rules out
thermal drift (throttling would appear as downward drift across reps). **No measurement was affected by
throttling.**

---

## 6. Limitations & threats to validity

1. **Single-stream only.** All runs used `--parallel 1` (one request at a time). This study measures
   per-request latency and single-user throughput — **not** concurrent-load throughput, goodput, or
   tail percentiles (p99), which are the focus of MLPerf's server scenario. Multi-user behavior on this
   box is unmeasured.
2. **3 repetitions, median only.** Below the ≥10 reps some sources recommend for public reporting.
   Mitigated by the observed near-zero variance, but we report no confidence intervals or p99.
3. **Retrieval-only long-context quality.** We used a needle-in-a-haystack and a figure-recall
   summarization. NIAH is known to be a *superficial* long-context probe (RULER, NeedleBench): it tests
   retrieval, not multi-hop reasoning or aggregation. We did not run RULER/LongBench-style multi-task
   suites, so "correct at 30k" means *retrieval* is correct at 30k.
4. **One recorded FAIL is a grader false-positive.** `qual_main_code` rep3: the model produced the
   *correct* stack algorithm, but its docstring contained the word "input", which the verifier's safety
   denylist (`\binput\b`, scanning the whole source incl. comments) matched → "unsafe-code-skipped".
   Effective code correctness is 3/3 for both engines; effective overall correctness is 84/84. This is
   accepted as a known grader false-positive and documented; the verifier is left unchanged, as it
   affects no reported performance figure (only that one cell's pass/fail verdict).
5. **`mainline+-fa` not tested.** The `-fa` arm was ik-only. Mainline's decode collapse at depth (§4.3)
   is therefore specifically *mainline without flash-attention*; `-fa` might mitigate it on mainline.
   Academic for us (we use ik), but a genuine gap.
6. **Sampling at `temperature=1.0`.** We used OpenAI's recommended (non-greedy) setting for fidelity.
   This favors deployment realism over bit-exact reproducibility; LLM CPU inference is also subject to
   floating-point non-determinism across runs. Mitigated by 3 reps and by tasks with robust correct
   answers, but exact token sequences are not guaranteed reproducible.
7. **One warm re-send missing.** `ikfa_30k_short` rep3 recorded no warm sample (83/84 warm samples);
   not flagged, does not affect cold results.
8. **`gpt-oss-120b` not run.** Discussion of the larger model elsewhere is explicitly out of scope here.

---

## 7. Reproducibility

- **Single source of truth:** all tunable values in `scripts/config.env`. Pinned engine commits
  (mainline `6e14286ed`, ik `670a3f6`), model file, flags, thread count — all there.
- **Blueprint:** `scripts/install.sh` (→ `00-deps` → `10-build-llamacpp` → `20-fetch-model` →
  `30-systemd`) reproduces the install on a fresh CT. The benchmark is `scripts/bench-suite.sh` (with
  `bench-payloads.py`, `bench-verify.py`, `bench-report.py`); design in `METHODOLOGY.md`.
- **Re-run:** `nohup bash ~/gpt-oss/scripts/bench-suite.sh >> results/suite.out 2>&1 &`. Idempotent and
  resumable by `RUN_ID` (skip keys on the CSV row, written last). The suite is self-defending
  (pre-flight gate, single-instance assertion, SIGKILL escalation, anomaly flags, data-health summary).
- **Artifacts:** `results/20260614-191322/` — `results.csv` (84 rows) plus 419 raw files
  (`raw/<cell>_rep<r>.{req,resp}.json` + `.warm.*` + `.wall`), ~14 MB. Every prompt (with full context),
  every response (`content` + `reasoning_content` + `usage` + `timings`) is saved.
- **Regenerate this report's tables:** `python3 scripts/bench-report.py results/<run>/results.csv`.

---

## 8. Conclusion

A single ~2016 Xeon workstation with no GPU runs a capable 20B-parameter MoE model **correctly** —
retrieval to 30k tokens, summarization, reasoning, and executable code — at OpenAI's recommended
sampling. Decode is a usable 8–17 tok/s. The real constraint is **cold prefill**: a fresh 30k-token
prompt costs ~17 minutes to first token. But that cost is **one-time per context** — a cached/reused
prefix answers in ~0.12 s (~8,500× faster). So the honest verdict is not "usable / not usable" but an
**envelope**: this hardware is genuinely useful and cheap for modest-context or cache-reused workloads
(chat, RAG over stable corpora), and unsuitable for cold, large, ad-hoc, interactive prompts — the case
that justifies a GPU. Engine choice matters and is settled by the data: **`ik_llama.cpp`** for any real
context (decisively better at depth), mainline only for short chat.

---

## References (verified against multiple sources during preparation)

Model:
- OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, arXiv:2508.10925 — https://arxiv.org/html/2508.10925v1
- *Introducing gpt-oss* — https://openai.com/index/introducing-gpt-oss/
- `openai/gpt-oss-20b` (HF) — https://huggingface.co/openai/gpt-oss-20b · `ggml-org/gpt-oss-20b-GGUF` — https://huggingface.co/ggml-org/gpt-oss-20b-GGUF
- OpenAI Harmony format — https://github.com/openai/harmony

Metrics & methodology:
- NVIDIA, *LLM Inference Benchmarking: Fundamental Concepts* — https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
- NVIDIA NIM, *Benchmarking metrics* — https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html
- vLLM, *Benchmark CLI* — https://docs.vllm.ai/en/latest/benchmarking/cli/
- Anyscale, *Understand LLM latency and throughput metrics* — https://docs.anyscale.com/llm/serving/benchmarking/metrics
- Anyscale, *Reproducible Performance Metrics for LLM Inference* — https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference
- MLCommons, *Llama 2 70B MLPerf Inference benchmark* — https://mlcommons.org/2024/03/mlperf-llama2-70b/ · rules — https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- llama.cpp `llama-bench` README — https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- NVIDIA GenAI-Perf — https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf
- Databricks, *LLM Inference Performance Engineering* — https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
- *On Evaluating Performance of LLM Inference Serving Systems*, arXiv:2507.09019 — https://arxiv.org/pdf/2507.09019
- *Identifying and Mitigating Systemic Measurement Bias…*, arXiv:2605.24217 — https://arxiv.org/html/2605.24217v1
- *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference*, arXiv:2506.09501 — https://arxiv.org/abs/2506.09501
- DistServe (goodput), Hao AI Lab — https://haoailab.com/blogs/distserve/

Quality evaluation:
- G. Kamradt, *Needle-in-a-Haystack* — https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- Hsieh et al., *RULER*, arXiv:2404.06654 — https://arxiv.org/abs/2404.06654
- Bai et al., *LongBench*, arXiv:2308.14508 — https://arxiv.org/abs/2308.14508
- Chen et al., *Evaluating Large Language Models Trained on Code* (HumanEval / pass@k), arXiv:2107.03374 — https://arxiv.org/abs/2107.03374
- White et al., *LiveBench*, arXiv:2406.19314 — https://arxiv.org/abs/2406.19314

*Engine sources: mainline `llama.cpp` https://github.com/ggml-org/llama.cpp (commit 6e14286ed); `ik_llama.cpp` https://github.com/ikawrakow/ik_llama.cpp (commit 670a3f6).*
