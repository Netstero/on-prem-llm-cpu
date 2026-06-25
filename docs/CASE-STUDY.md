# Running capable LLMs on a 2016 CPU workstation: a two-model case study

**How cheaply can you run a sensible LLM on-premises, CPU-only — with receipts.**

This report combines two measured studies run on the *same* hardware, with the *same* harness and
methodology:
- **`openai/gpt-oss-20b`** (MoE, MXFP4) — run `20260614-191322`, written up in full in **`gpt-oss-20b.md`**.
- **`Qwen3-Coder-30B-A3B-Instruct`** (MoE, UD-Q4_K_XL) — run `20260618-220001`, written up in full in
  **`qwen3-coder-30b.md`**.

Each study is 84 measured runs (28 cells × 3 reps); **168 runs total**, every raw artifact saved. This
document presents the shared methodology once, both models' results, and a head-to-head comparison. The
two per-model documents remain the authoritative sources; nothing here is re-estimated — figures are
medians of 3 reps taken verbatim from each run's `results.csv`, and derived figures (ratios) are marked.

| | gpt-oss-20b | Qwen3-Coder-30B-A3B |
|---|---|---|
| Run ID | `20260614-191322` | `20260618-220001` |
| Date | 2026-06-15 | 2026-06-18/19 |
| Runs | 84 (28 cells × 3 reps) | 84 (28 cells × 3 reps) |
| Effective correctness | 84/84 (one FAIL = grader false-positive, §8) | valid cells pass except summarization (§8); deepest cells invalid (§5.6) |
| Thermal | 0 throttle events (gpt-oss study, §7.4) | not separately instrumented this run; same box/limits |

---

## 1. Summary of findings

1. **Both models run CPU-only on this box and are useful.** gpt-oss-20b solves every task at OpenAI's
   recommended sampling. Qwen3-Coder passes retrieval, reasoning, and code; it under-performs on the
   figure-recall summarization task (§5.5, §8).
2. **Qwen decodes *faster* than gpt-oss at short/medium context.** Short-context decode (ik): Qwen ~22
   tok/s vs gpt-oss ~13.6 tok/s. A pre-test desk estimate of 5–11 tok/s for Qwen was too pessimistic.
3. **gpt-oss prefills better at depth, so its cold long-prompt latency is lower.** At the ~16k cell (ik),
   gpt-oss prefill ~51 t/s vs Qwen ~26 t/s → cold TTFT ~308 s vs ~697 s. There is a **decode crossover**
   around 16k where gpt-oss edges ahead (§6).
4. **Engine choice — `ik_llama.cpp` is the default for both, decisively at depth.** For gpt-oss, mainline
   `llama.cpp` wins only short-prompt chat decode (+22%); ik wins everywhere else and holds decode at
   depth where mainline collapses (~3× at 30k). For Qwen, **ik wins or ties at every valid depth** — a
   cleaner verdict (16k decode 2.6× mainline).
5. **Flash-attention (`-fa`) makes no measurable difference** for either model at the depths tested — a
   clean negative result, twice.
6. **Cold prefill is the wall, for both.** First-token latency on a fresh large prompt is minutes on CPU;
   the same prompt against a warm prefix cache returns in a fraction of a second (≈0.1 s with ik; up to
   ~1.5 s for mainline gpt-oss at 30k). The "minutes to first token" cost is per-fresh-context, not per-request.
7. **A methodological result (Qwen):** gpt-oss-calibrated test fixtures *overflow* Qwen's context — the
   "30k" cell tokenizes to 34,483 tokens under Qwen's tokenizer, over the 32,768 window the run used.
   Mainline rejected it correctly; ik silently truncated. **All Qwen "30k" cells are excluded** (§5.6).

---

## 2. System under test

### 2.1 Hardware (shared; ~2016-era, used-market class)
| Component | Spec |
|---|---|
| Machine | Dell Precision T5810 workstation |
| CPU | Intel Xeon E5-2680 v4 (Broadwell, 14C/28T, base 2.4 GHz, AVX2; **no** AVX-512/AMX) |
| Memory | 64 GB DDR4-2400, quad-channel (4×16 GB); theoretical bandwidth ≈76.8 GB/s, effective ~55–60 GB/s (estimated, not measured) |
| GPU | none (CPU-only) |
| Storage | NVMe SSD |

### 2.2 Container & isolation (shared)
- Unprivileged Proxmox LXC (Debian 13), CTID <CTID>.
- **CPU pinning:** host `cpuset` 0–11 → 12 distinct physical cores (HT siblings in 14–25; no collisions).
- Memory cgroup cap: **24 GB** at the gpt-oss run; raised to **~31 GB** before the Qwen run (Qwen weights
  ~16.4 GiB + KV need more headroom). Inference ran as the **sole** model process throughout (§3.5).
- Threads: **11** of the 12 pinned cores (one reserved for OS/sshd/monitoring; rationale §7.3).

### 2.3 Models

**2.3.1 gpt-oss-20b** (OpenAI model card, arXiv:2508.10925; HF `config.json`):
- **20.9B** total / **3.6B active** per token; **32 experts, top-4**; **24 layers**.
- **Native MXFP4** quant of MoE expert weights (~4.25 bpp); other tensors BF16. GGUF
  `gpt-oss-20b-mxfp4.gguf` = **12.1 GB** (~11.27 GiB).
- Context **131,072**; tokenizer `o200k_harmony` (vocab 201,088); **Harmony** chat format mandatory.
  License **Apache-2.0**; released **2025-08-05**.
- Sampling used: OpenAI-recommended **`temperature=1.0, top_p=1.0`** (`top_k=0, min_p=0`), no repeat
  penalty, no KV-cache quant.

**2.3.2 Qwen3-Coder-30B-A3B-Instruct** (values read from the loaded GGUF; license from model card):
- **30.532B** total / **~3.3B active** per token (model card); **128 experts, top-8**; **48 layers**;
  GQA **32 query / 4 KV heads**, head dim **128**, embedding dim **2048**.
- Context **262,144** (256K native); tokenizer BPE, vocab **151,936**.
- Quant **UD-Q4_K_XL** (Unsloth Dynamic; GGUF ftype `Q4_K - Medium`); model size **16.447 GiB**
  (4.627 bpw); **17,665,334,432 B** on disk (byte-exact to the HF LFS size). License **Apache-2.0**.
- Sampling: the server was set to Qwen's recommended **`temperature=0.7, top_p=0.8, top_k=20, min_p=0,
  repeat_penalty=1.05`**, no KV-cache quant — **but a harness defect overrode it**. The request body
  hard-codes `temperature=1.0, top_p=1.0, top_k=0, min_p=0`, which llama.cpp honors over the server flags,
  so the Qwen runs **actually sampled at temp 1.0 / top_p 1.0 / top_k 0 / min_p 0** (`repeat_penalty 1.05`
  did apply — it is absent from the body). Speed metrics are sampling-independent and unaffected; only
  quality grading is — the summarization result was generated at temp 1.0 (§8, item 11). gpt-oss intended
  exactly the body's values, so it is unaffected.

### 2.4 Engines (shared)
- **mainline `llama.cpp`** — commit `6e14286ed` (`llama-server` build b9631). Built
  `cmake -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON` (native arch auto-detected → AVX2/FMA).
- **`ik_llama.cpp`** — commit `670a3f6`, a CPU/MoE-optimized fork. Built with explicit
  `-DGGML_NATIVE=ON -DGGML_AVX2=ON -DGGML_FMA=ON`. **Default engine.**
- Common serve flags: `--ctx-size 32768 --parallel 1 --jinja --threads 11 --threads-batch 11`, no
  repetition penalty (gpt-oss), no KV-cache quant. Variant **`ikfa`** = ik + `-fa 1` (flash attention),
  applied to long-context cells only. Both engines load the `qwen3moe` and gpt-oss architectures without
  error.

---

## 3. Methodology (identical harness for both studies)

### 3.1 Metrics and conventions
We report the two physically distinct phases of inference separately — **prefill** (prompt processing,
compute-bound) and **decode** (token generation, memory-bandwidth-bound). All values come from
`llama-server`'s own `timings`/`usage` JSON plus client wall-clock:

| Metric (column) | Definition | Maps to |
|---|---|---|
| `prefill_tps` | `timings.prompt_per_second` | prefill throughput |
| `decode_tps` | `timings.predicted_per_second` (decode phase, excl. first token) | inverse of TPOT |
| `ttft_s` | `timings.prompt_ms / 1000` | **cold TTFT** |
| `warm_ttft_s` | same, for the warm re-send (cached prefix) | warm TTFT |
| `wall_s` | client `curl` `time_total` | end-to-end latency |
| `prompt_tokens` / `completion_tokens` | from `usage` | input depth / output produced |

The client runs on the same host (`curl` → `127.0.0.1`), so network latency ≈ 0 and `ttft_s` is a tight
lower bound on observed TTFT. `decode_tps` excludes the first token (vLLM / NVIDIA GenAI-Perf convention).
For gpt-oss, `completion_tokens` includes the Harmony reasoning channel (the honest decode workload).

### 3.2 Experimental design (the grid)
A **cell** is one exact combination we measure (e.g. *ik engine, ~16k-token input, short output*). Each
cell is run 3 times and summarized by the median, so **runs = cells × 3**. Three groups, per model:

| group | what is varied | cells | runs (×3) |
|---|---|---|---|
| **Speed** | 2 engines (mainline, ik) × 4 input sizes (~600, ~4k, ~16k, ~30k tok) × 2 output lengths (short ≤64 tok, long ≤2000 tok) | 16 | 48 |
| **Quality** | 2 engines × 4 tasks (retrieval@16k, summarization@16k, reasoning, coding) | 8 | 24 |
| **Flash-attention** | ik + `-fa` only, at the 2 deep inputs (16k, 30k) × 2 output lengths | 4 | 12 |
| | | **28** | **84** |

Input sizes were set by a deterministic filler-doc generator (~30.5 tokens/section under gpt-oss's
tokenizer). The two models tokenize the same fixtures differently (different tokenizers *and* chat
templates), so realized token depths differ per cell: at the deep cells Qwen runs ~12–13% higher
(16k: 17,873 vs 15,960; 30k: 34,483 vs 30,554), which caused the overflow (§5.6); at the smallest cell it
is slightly *lower* (600: 599 vs 638). See §6.

### 3.3 Per-run protocol (what makes the numbers credible)
Each measured run has **three setup steps that record nothing**, then **two measurements that do** — the
numbers in `results.csv` come *only* from the two measurement sends, never from the warm-up — then a save:

*Setup (not recorded):*
1. **Fresh server process** launched → cold KV cache + cold prompt cache. A new process per run (incl. each
   of the 3 reps) is the isolation (§3.5).
2. **Throwaway warm-up — not a measurement.** A *different*, tiny prompt is sent and discarded, only to
   absorb one-time startup cost; being a different prompt, it leaves the test prompt uncached (still cold).

*Measurements (recorded):*
3. **Cold send — the main measurement.** The test prompt, first time, via `curl` → `127.0.0.1` → cold
   prefill, decode, **cold TTFT**.
4. **Warm send — the same prompt again.** Identical prompt re-sent without restart (`max_tokens=16`) → a
   prefix-cache hit → `warm_prefill_tps` / **`warm_ttft_s`**.

*Save:*
5. **Capture & grade**, append one CSV row, save raw request/response JSON for both sends, kill the server.

The point of steps 3 vs 4: the same prompt yields a **cold** number (first ask, prefill-dominated) and a
**warm** number (cached, near-instant). That cold-vs-warm contrast is a headline of both studies.

### 3.4 Quality verification (so speed isn't bought with garbage)
Every task is **deterministically auto-graded** — no LLM-as-judge, no string-similarity:

| Task | Design | Grading | Lineage |
|---|---|---|---|
| Retrieval (needle) | benign fact (jersey number `4417`) planted at 55% depth in a 16k filler doc | exact-substring `4417` | Needle-in-a-Haystack (Kamradt, 2023) |
| Summarization | doc with **8 sentinel figures**; ask for a summary "including every figure" | recall of the 8 figures; PASS ≥ 0.75 | numeric-recall |
| Reasoning | multi-step arithmetic (answer 53), output ends `ANSWER: <n>` | last `ANSWER:` == 53 | exact-match (cf. GSM8K) |
| Coding | write `is_balanced(s)` for ()[]{} | **execute** the code vs 8 cases incl. `([)]`; PASS = 8/8 | functional correctness (HumanEval) |

The needle is a benign jersey number (a "secret/code" framing triggers gpt-oss refusals; odd names get
normalized by the tokenizer). Every speed run also embeds the needle, so a fast-but-broken run can't pass.

### 3.5 Isolation & validity controls
- **Cold-cache guarantee:** fresh process per run; re-sending an identical prompt to a live server is a
  prefix-cache hit (the most common way benchmarks are inflated) — so the warm-up uses a *different*
  prompt and the warm re-send is measured and labelled *separately*.
- **Single instance enforced** before each measurement (clean + retry once, else flag).
- **Pre-flight gate:** aborts if the box isn't clean (stray server, busy port, <15 GB RAM).
- **Anomaly flags:** multi-instance / low-mem / parse-error / implausibly-low cold TTFT. *gpt-oss run: no
  run flagged. Qwen run: no run flagged* (the overflow in §5.6 is a context-limit error, a category the
  flag set did not cover — noted as a harness gap).
- **Thermal** monitored on the gpt-oss run (§7.4).

### 3.6 Statistics
3 reps per cell, summarized by **median**. Single-stream study (`--parallel 1`): no concurrent load, no
p99/goodput (see §8).

### 3.7 How this methodology compares to industry best practice
Benchmarked against MLPerf Inference, vLLM `benchmark_serving`, llama.cpp `llama-bench`, NVIDIA
GenAI-Perf, and the LLM-eval literature (NIAH/RULER, HumanEval/pass@k, goodput). Aligned: warm-up;
cold/warm distinction reported as *both*; prefill/decode separation; TPOT excludes first token; sweep
input AND output; full HW/SW/commit/flags/sampling disclosure; quality reported alongside speed;
deterministic auto-grading incl. code execution; single-instance + pinning + thermal logging. Partial /
out of scope: **3 reps** (some sources advise ≥10; mitigated by near-zero variance); **no p99 / no
concurrency / no goodput** (single-stream by design); retrieval-style long-context only (no RULER
multi-hop); non-greedy sampling (fidelity over bit-reproducibility). Net: consistent with serving
best-practice on the axes relevant to a **single-user, on-prem, CPU** deployment; it deliberately omits
the concurrent-load axis.

---

## 4. Results — gpt-oss-20b (run `20260614-191322`)

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

### 4.2 Cold vs warm TTFT (ik) — the central result
| input | TTFT cold | TTFT warm | ratio (derived) |
|---|---|---|---|
| 638 | 6.76 s | 0.07 s | ~97× |
| 4104 | 50.59 s | 0.08 s | ~632× |
| 15960 | 308.2 s | 0.10 s | ~3,080× |
| 30554 | 1026.6 s | 0.12 s | **~8,555×** |

### 4.3 Engine comparison
Prefill (ik advantage, derived): 638 **1.66×**, 4104 **1.68×**, 15960 1.38×, 30554 **1.02× (converges)**.
Decode (short cells): 638 mainline **+22%** (16.64 vs 13.60); 4104 ≈ tie; 15960 ik **1.54×** (10.90 vs
7.10); 30554 ik **~2.97×** (8.72 vs 2.94). **Mainline decode collapses with depth; ik degrades
gracefully.**

### 4.4 Flash-attention (ik vs ik+`-fa`)
Differences within run-to-run noise (<1%) at 16k and 30k. **No measurable effect.** (`mainline+-fa` not
tested — §8.)

### 4.5 Quality (medians; all auto-graded)
| engine | task | input tok | prefill t/s | decode t/s | TTFT | correct |
|---|---|---|---|---|---|---|
| ik | retrieval@16k | 15960 | 51.38 | 11.01 | 309.4 s | **3/3** |
| ik | summarization@16k | 16039 | 51.28 | 10.57 | 311.6 s | **3/3** (8/8 figures) |
| ik | reasoning | 146 | 68.50 | 13.60 | 1.23 s | **3/3** |
| ik | coding | 142 | 65.70 | 13.54 | 1.22 s | **3/3** |
| main | retrieval@16k | 15960 | 37.53 | 6.88 | 423.7 s | **3/3** |
| main | summarization@16k | 16039 | 37.30 | 6.70 | 428.4 s | **3/3** (8/8 figures) |
| main | reasoning | 146 | 50.19 | 17.15 | 1.71 s | **3/3** |
| main | coding | 142 | 49.41 | 17.28 | 1.66 s | 2/3 (1 false-pos, §8) |

gpt-oss is **correct on every task**, both engines; the difference between engines is purely speed.

---

## 5. Results — Qwen3-Coder-30B-A3B (run `20260618-220001`)

Cells are matched to gpt-oss by *fixture* (section count), but the two models tokenize the same text
differently, so actual depths differ (`≈Qwen tok` column; Qwen runs higher at the deep cells, slightly
lower at the smallest — §3.2). All values are medians of 3 reps. **The "30k" cells are excluded as
invalid — see §5.6.**

### 5.1 Speed — decode (tok/s), valid cells
| cell (≈Qwen tok) | ik short | ik long | main short | main long | ikfa short | ikfa long |
|---|---|---|---|---|---|---|
| 600 (~0.6k) | **22.49** | **17.13** | 22.06 | 15.17 | — | — |
| 4k (~4.4k) | **17.43** | **12.95** | 11.80 | 8.71 | — | — |
| 16k (~17.9k) | **9.09** | **6.93** | 3.42 | 2.56 | 9.20 | 6.79 |

### 5.2 Speed — prefill (tok/s), valid cells
| cell | ik short | ik long | main short | main long | ikfa short | ikfa long |
|---|---|---|---|---|---|---|
| 600 | **97.25** | **97.64** | 58.62 | 58.50 | — | — |
| 4k | **65.01** | **63.97** | 40.80 | 40.86 | — | — |
| 16k | **25.66** | **25.44** | 18.94 | 18.90 | 25.16 | 24.91 |

### 5.3 Cold vs warm TTFT (seconds), valid cells
| cell | ik cold | ik warm | main cold | main warm | ikfa cold | ikfa warm |
|---|---|---|---|---|---|---|
| 600 (short/long) | 6.16 / 6.53 | ~0.06 | 10.20 / 10.89 | ~0.06 | — | — |
| 4k | 67.34 / 69.05 | ~0.07 | 107.27 / 108.07 | ~0.11 | — | — |
| 16k | 696.59 / 704.16 | ~0.14 | 943.47 / 947.71 | ~0.37 | 710.25 / 719.04 | ~0.14 |

### 5.4 Flash-attention (ik vs ik+`-fa`, 16k cell)
decode 9.09 vs 9.20, prefill 25.66 vs 25.16, cold TTFT 696.59 vs 710.25 s — **no useful difference**
(consistent with gpt-oss).

### 5.5 Quality (medians; all auto-graded)
| task | ≈tokens | ik decode | ik TTFT | ik result | main decode | main TTFT | main result |
|---|---|---|---|---|---|---|---|
| retrieval (needle) | 17,873 | 9.05 | 703.65 s | **PASS** (1.0) | 3.44 | 943.10 s | **PASS** (1.0) |
| summarization | 17,965 | 7.10 | 724.67 s | **FAIL** (0.375) | 2.67 | 959.22 s | **FAIL** (0.375) |
| reasoning | 88 | 18.73 | 1.04 s | **PASS** (1.0) | 20.21 | 1.59 s | **PASS** (1.0) |
| coding | 80 | 18.88 | 0.94 s | **PASS** (1.0) | 20.34 | 1.39 s | **PASS** (1.0) |

Retrieval, reasoning, and code pass on both engines. **Summarization scored 3/8 figures (0.375)**
identically across both engines and all reps — systematic (§8). Note that on the *same* task and grader
gpt-oss scored 8/8 (§4.5), so the grader is functional. Two non-exclusive explanations for Qwen's 3/8
remain (not disambiguated this run): this run sampled at temperature 1.0 rather than the intended 0.7
(§2.3.2 / §8 item 11) — which plausibly hurts exact figure recall — and/or Qwen omits/reformats figures.

### 5.6 The "30k" cells are invalid (context overflow) — primary Qwen caveat
The run used `--ctx-size 32768` (to mirror the gpt-oss study). The "30k" fixture tokenizes to **34,483
tokens** under Qwen's tokenizer — beyond the 32,768 window. The engines diverged:
- **mainline `llama.cpp`** returned `HTTP 400 — "request (34483 tokens) exceeds the available context size
  (32768 tokens)"` → 6 honest FAIL rows.
- **`ik_llama.cpp`** did *not* error; it truncated/shifted and reported `prompt_tokens ≈ 18,099` (not
  34,483) → PASS rows that are **not** a real 30k-token decode (tell: its "30k" depth ≈ its "16k" depth).

**All "30k" cells (ik, ikfa, main) are excluded.** The deepest valid Qwen measurement is the "16k" cell
(~17.9k tokens). This reflects mismatched tokenizers, not the model — Qwen natively supports 262,144
tokens, and the box has RAM for a 64k window; a true deep-context characterization needs a
Qwen-tokenizer-calibrated fixture, a larger `--ctx-size`, and an overflow guard in the harness.

---

## 6. Head-to-head comparison

Decode (tok/s), **ik_llama.cpp**, medians. Depths are *not* token-identical (at these cells gpt-oss
tokenizes lower — e.g. its "16k" cell ≈ 15,960 tok vs Qwen's ≈ 17,873 tok), so read as comparable-depth:

| cell | gpt-oss · short | gpt-oss · long | Qwen · short | Qwen · long |
|---|---|---|---|---|
| 600 | 13.60 | 13.21 | **22.49** | **17.13** |
| 4k | 12.81 | 12.47 | **17.43** | 12.95 |
| 16k | **10.90** | **10.22** | 9.09 | 6.93 |

- **Prefill (ik):** gpt-oss 85 / 80 / 51 vs Qwen 97 / 65 / 26 t/s (at 600 / 4k / 16k). Qwen prefills
  faster at the smallest prompt but **degrades faster with depth**; gpt-oss holds prefill better.
- **Cold TTFT at 16k (ik):** gpt-oss ~308 s vs Qwen ~697 s — gpt-oss reaches first token ~2.3× sooner on
  a comparable deep prompt.
- **Crossover:** Qwen decodes faster at short/medium context; around the 16k cell gpt-oss edges ahead on
  decode (partly because Qwen's cell is ~12% deeper in tokens) and prefills markedly better.
- **Engine verdict differs by model:** for gpt-oss, mainline wins short-prompt decode (+22%) and ik wins
  at depth; for Qwen, **ik wins or ties at every valid depth** (mainline only narrowly leads on the
  tiniest reasoning/code prompts), so ik is the cleaner choice. Above ~32k no head-to-head is possible on
  this box (gpt-oss was studied at 32k ctx; Qwen overflowed — §5.6).

---

## 7. Discussion

### 7.1 The usable envelope (both models)
Usability on this box is set by **prefill / cold TTFT**, not decode. Cold first-token latency (ik):

| context | gpt-oss cold TTFT | Qwen cold TTFT | interactively usable? |
|---|---|---|---|
| ~600 tok | ~7 s | ~6 s | yes |
| ~4k tok | ~51 s | ~67 s | borderline |
| ~16–18k tok | ~5 min | ~12 min | only with caching / pre-warm |
| ~30k tok | ~17 min (gpt-oss) | n/a (overflow) | only with caching / pre-warm |

Because the cold cost is one-time (warm TTFT ≈0.06–0.14 s with ik regardless of depth; up to ~1.56 s for
mainline gpt-oss at 30k), the usable pattern for large context is: prefill the stable context once (off
the critical path), then operate over it warm. Both
models are useful and cheap for modest-context or cache-reused workloads (chat, RAG over stable corpora);
neither is suited to cold, large, ad-hoc, interactive prompts — the case that justifies a GPU.

### 7.2 Engine recommendation
- **gpt-oss:** ik for anything with real context (≥4k), mainline only for pure short chat (its ~22%
  short-context decode edge). Default: **ik**.
- **Qwen:** **ik** across the board — faster or tied at every valid depth, decisively so beyond ~4k.
- ik is also the leaner-memory engine (mmap-shared model). The standing default for both is **ik**.

### 7.3 Why 11 threads
Decode is memory-bandwidth-bound and saturates near 10 cores; prefill is compute-bound and scales with
cores. Using all 12 starves the container's control plane (SSH lockouts observed at 12). `--threads 11`
keeps ~full prefill, ~95% of peak decode, and a responsive box. Served decode rates are below the
synthetic `llama-bench` ceiling because the HTTP + chat-template + sampling path has real overhead —
a reason to benchmark the *served* path.

### 7.4 Thermal validity (gpt-oss run)
The per-package throttle counter read **0 before and after** the multi-hour gpt-oss run; idle 49 °C, peak
~77 °C; near-zero rep-to-rep variance rules out thermal drift. The Qwen run was not separately
instrumented but used the same box, limits, and thread count.

---

## 8. Limitations & threats to validity

1. **Single-stream only** (`--parallel 1`) for both: per-request latency / single-user throughput — not
   concurrent throughput, goodput, or p99.
2. **3 repetitions, median only** — below the ≥10 some sources advise; mitigated by near-zero variance.
3. **Retrieval-only long-context quality** (needle + figure-recall); not RULER/LongBench multi-hop.
4. **Qwen "30k" cells invalid** (context overflow, §5.6) — deepest valid Qwen depth ~18k; no Qwen
   deep-context (64k/256k) measured. Harness gap: the flag set did not catch context overflow / silent
   truncation; add an overflow guard before any deep re-run.
5. **Qwen summarization scored 0.375** on both engines (3/8 figures), systematically. The same grader
   passed gpt-oss at 8/8, so it is functional. Two non-exclusive causes, **not disambiguated**: this run
   sampled at temperature 1.0 not the intended 0.7 (item 11 / §2.3.2), which plausibly hurts figure
   recall; and/or Qwen omits/reformats figures (raw summary not inspected). Affects only the summarization
   quality verdict, not any speed metric.
6. **gpt-oss one recorded FAIL is a grader false-positive** (`qual_main_code` rep3): correct code, but a
   docstring containing "input" tripped the verifier's safety denylist. Effective code correctness 3/3;
   accepted as a known false-positive, verifier left unchanged (affects no performance figure).
7. **`mainline+-fa` not tested** (the `-fa` arm was ik-only) — mainline's decode collapse at depth is
   specifically *mainline without flash-attention*.
8. **Non-identical token depths across models** (different tokenizers on the same fixtures) make the
   head-to-head comparable-depth, not token-matched.
9. **Effective memory bandwidth not directly measured**; only the 76.8 GB/s theoretical peak is stated.
10. **One Qwen warm re-send and one gpt-oss warm re-send** edge cases noted in the per-model docs; neither
    affects cold results.
11. **Qwen sampling override (harness defect, corrected post-run).** The payload generator hard-coded
    `temperature 1.0, top_p 1.0, top_k 0, min_p 0` into every request body, which llama.cpp honors over the
    server's `--temp 0.7 / top_p 0.8 / top_k 20` flags — so the Qwen runs sampled at 1.0, not 0.7
    (`repeat_penalty 1.05`, absent from the body, did apply). Verified via the server's `/props`, the saved
    request bodies, and a greedy probe (`temperature:0, top_k:1` → byte-identical output). **Speed metrics
    are sampling-independent and unaffected;** only the summarization quality result is implicated (item 5).
    gpt-oss is unaffected (its intended sampling equals the body's values). Generator since fixed to defer
    to the server flags.

---

## 9. Reproducibility
- **Single source of truth:** `scripts/config.env` (pinned engine commits mainline `6e14286ed`, ik
  `670a3f6`; model files; flags; `ACTIVE_MODEL` toggle + `QWEN_*` vars; thread count).
- **Blueprint:** `scripts/install.sh` (`00-deps` → `10-build-llamacpp` → `20-fetch-model` → `30-systemd`).
  Benchmark: `scripts/bench-suite.sh` (+ `bench-payloads.py`, `bench-verify.py`, `bench-report.py`);
  design in `METHODOLOGY.md`. Re-run: `nohup bash scripts/bench-suite.sh >> results/suite.out 2>&1 &`
  (idempotent, resumable by `RUN_ID`).
- **Artifacts:** gpt-oss `results/20260614-191322/` (84 rows + 419 raw files, ~14 MB);
  Qwen `results/20260618-220001/` (84 rows + raw). Every prompt and full response (`content`,
  `reasoning_content` where applicable, `usage`, `timings`) saved.
- **Full per-model write-ups:** `gpt-oss-20b.md` (gpt-oss) and `qwen3-coder-30b.md` (Qwen) — the authoritative
  sources this document combines.

---

## 10. Conclusion
A single ~2016 Xeon workstation with no GPU runs **two** capable MoE models usefully. gpt-oss-20b is
correct across every task; Qwen3-Coder-30B-A3B is faster at short/medium context (and a coding-specialist)
but weaker on the figure-recall summarization task and bounded here to ~18k tokens by a tokenizer/context
mismatch. For both, **decode is usable (≈3–22 tok/s depending on depth) and cold prefill is the wall** —
minutes to first token on a fresh large prompt, collapsing to ~0.1 s (ik) when the context is cached. **`ik_llama.cpp` is the engine of choice for both** (decisively at depth). The honest verdict is
an *envelope*: this hardware is genuinely useful and cheap for modest-context or cache-reused workloads,
and unsuited to cold, large, ad-hoc, interactive prompts — the case that justifies a GPU.

---

## References
Full reference lists (model cards, metrics/methodology, quality-eval lineage) are in the per-model
documents: **`gpt-oss-20b.md` §References** (gpt-oss; NVIDIA/vLLM/MLPerf/Anyscale metrics, NIAH/RULER/
HumanEval/LiveBench, arXiv:2508.10925) and **`qwen3-coder-30b.md`** (Qwen3-Coder model card / Unsloth GGUF;
methodology shared with the gpt-oss study). Engine sources: mainline `llama.cpp`
https://github.com/ggml-org/llama.cpp (commit `6e14286ed`); `ik_llama.cpp`
https://github.com/ikawrakow/ik_llama.cpp (commit `670a3f6`).
