# Qwen3-Coder-30B-A3B on a CPU-only on-prem box — a measured case study

**Run ID:** `20260618-220001`  ·  **Date:** 2026-06-18/19  ·  **Data:** `results/20260618-220001/results.csv` (84 rows)
**Companion study:** `gpt-oss-20b.md` (gpt-oss-20b on the same box, run `20260614-191322`). This document reuses
that study's harness and methodology unchanged; for the full benchmarking-best-practice discussion see
`gpt-oss-20b.md` §3 and `METHODOLOGY.md`. Here we report Qwen-specific results and a like-for-like comparison.

---

## 1. Summary of findings

- **ik_llama.cpp is the correct engine for Qwen on this box, at every tested depth.** At the ~18k-token
  cell, ik decodes **2.6×** faster than mainline llama.cpp (9.1 vs 3.4 tok/s) and prefills **1.36×** faster.
  At short context the two are close on decode but ik prefills ~1.7× faster. The gap widens with depth.
  This is a cleaner verdict than gpt-oss, where mainline won short-context decode.
- **Qwen3-Coder is faster than a pre-test estimate suggested.** Short-context decode reaches ~22 tok/s
  (ik), *faster* than gpt-oss-20b on the same box (~13.6 tok/s). A 3.3B-active MoE at 4-bit is well-suited
  to this hardware. A prior desk estimate of 5–11 tok/s was too pessimistic.
- **Flash-attention (`-fa`) makes no useful difference** at the depths it was tested (ik vs ik+fa at ~18k:
  decode 9.1 vs 9.2, prefill 25.7 vs 25.2, cold TTFT 697s vs 710s). Consistent with the gpt-oss finding.
- **Cold first-token latency at depth is severe on CPU.** Cold TTFT at the ~18k cell is ~12 min (ik) /
  ~16 min (mainline); warm (prefix-cache hit) is ~0.14s. Prefill is compute-bound and is the practical
  bottleneck for long prompts.
- **A methodological result: gpt-oss-calibrated test fixtures overflow Qwen's context.** The deepest cell
  ("30k") tokenizes to **34,483 tokens** under Qwen's tokenizer — over the 32,768-token window the run was
  configured with. Mainline rejected it correctly (HTTP 400); ik silently truncated. **All "30k" cells are
  therefore excluded as invalid.** The deepest valid measurement is the "16k" cell (~17.9k tokens). See §6.1.

---

## 2. System under test

### 2.1 Hardware (spec)
- **Machine:** Dell Precision T5810, single-socket.
- **CPU:** Intel Xeon E5-2680 v4 — **14 physical cores / 28 threads**, Broadwell microarchitecture,
  **AVX2** (no AVX-512, no AVX-VNNI). Base 2.4 GHz.
- **Memory:** 64 GB DDR4-2400, **quad-channel**. Theoretical peak bandwidth
  4 × 2.4 GT/s × 8 B = **76.8 GB/s** (effective sustained is lower; not directly measured here).
- **Accelerator:** none — **CPU-only inference**.
- **Container:** unprivileged Proxmox LXC (CTID <CTID>), Debian. RAM cgroup cap ≈ 31 GB at test time;
  root disk 59 GB. Host-side CPU pinning to cores 0–11.

CPU inference of this model class is **memory-bandwidth-bound** for token generation (decode) and
**compute-bound** for prompt processing (prefill).

### 2.2 Model (ground truth, read from the loaded GGUF)
| Property | Value | Source |
|---|---|---|
| Name | `Qwen3-Coder-30B-A3B-Instruct` | GGUF `general.name` |
| Architecture | `qwen3moe` (sparse MoE) | GGUF `general.architecture` |
| Total parameters | **30.532 B** | GGUF `model params` |
| Experts / used per token | **128 / 8** | GGUF `n_expert` / `n_expert_used` |
| Active params per token | ~3.3 B | model card (derived from 8/128 experts) |
| Layers | **48** | GGUF `n_layer` / `block_count` |
| Attention heads (Q / KV) | **32 / 4** (GQA) | GGUF `n_head` / `n_head_kv` |
| Head dim; embedding dim | **128**; **2048** | GGUF `n_embd_head_k`; `n_embd` |
| Native context length | **262144** (256K) | GGUF `n_ctx_train` |
| Vocab | BPE, **151936** | GGUF `n_vocab` |
| Quantization | **UD-Q4_K_XL** (Unsloth Dynamic); ftype `Q4_K - Medium` | GGUF `model ftype` |
| Model size | **16.447 GiB** (4.627 bpw) on load; **17,665,334,432 B** on disk | GGUF `model size`; `stat` |
| License | Apache-2.0 | model card |

The on-disk size is byte-exact to the Hugging Face LFS size (`unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`),
verified at download.

### 2.3 Sampling — important correction
The server was started with Qwen's recommended sampling (`temperature 0.7`, `top_p 0.8`, `top_k 20`,
`min_p 0.0`, `repeat_penalty 1.05`, `--jinja`). **However, the benchmark requests did not use those
values.** The payload generator hard-codes `temperature 1.0, top_p 1.0, top_k 0, min_p 0.0` into every
request body, and llama.cpp lets request-body sampling override the server's CLI flags — so the Qwen runs
**actually sampled at `temperature 1.0, top_p 1.0, top_k 0, min_p 0.0`** (with `repeat_penalty 1.05`, which
was absent from the body and so kept its server value). This harness defect was found after the run (§6.4).
**Impact is confined to quality grading — prefill/decode/TTFT are sampling-independent, so every speed
figure in this report stands unchanged.** The summarization result (§4.4 / §6.2) was generated at
temperature 1.0, not 0.7. (The live server, queried without body sampling, does use 0.7 — verified.)

### 2.4 Engines
- **ik_llama.cpp** — commit `670a3f6` (built & verified 2026-06-14). Default engine.
- **mainline llama.cpp** — commit `6e14286ed`, `llama-server` build `b9631` (2026-06-14).
- Both invoked with: `--ctx-size 32768 --parallel 1 --threads 11 --threads-batch 11 --jinja`,
  **no KV-cache quantization**. Variant **`ikfa`** = ik_llama.cpp + `-fa 1` (flash attention), applied to
  long-context cells only.

Both engines support the `qwen3moe` architecture and loaded the model without error.

---

## 3. Methodology (summary; full rationale in `gpt-oss-20b.md` §3 / `METHODOLOGY.md`)

The harness (`scripts/bench-*`) is unchanged from the gpt-oss study. Each measured run has **three setup
steps that record nothing**, then **two measurements that do** (the numbers come *only* from steps 3–4,
never from the warm-up), then a save:

*Setup (not recorded):*
1. **Fresh server process** started → cold KV cache + cold prompt cache (process-level isolation, no reboot).
2. **Throwaway warm-up — not a measurement.** A *different*, tiny prompt is sent and discarded, only to
   absorb one-time startup cost. Being a different prompt, it leaves the test prompt uncached (still cold).

*Measurements (recorded):*
3. **Cold send — the main measurement.** The test prompt, sent for the *first* time via `curl` to
   `127.0.0.1` (no SSH in the request path) → cold prefill, decode, **cold TTFT**. Raw request/response,
   content, and `timings`/`usage` saved.
4. **Warm send — the same prompt again.** The *identical* prompt re-sent immediately, no restart
   (`max_tokens=16`) → a prefix-cache hit → **warm TTFT** / warm prefill.

*Save:*
5. **Grade** correctness (deterministic verifier), append one row to `results.csv`, then kill the server.

The point of steps 3 vs 4: the *same* prompt yields a **cold** number (first ask) and a **warm** number
(cached). **Metrics** (from server `timings`): **prefill t/s** = `prompt_per_second`; **decode t/s** =
`predicted_per_second`; **TTFT (cold)** = `prompt_ms/1000`; **TTFT (warm)** likewise on the warm send;
**wall s** = total request time. **Reps = 3**, summarized by **median** (robust to a single outlier).

**Design.** A **cell** is one exact combination (e.g. *ik, ~16k input, short output*); runs = cells × 3.
Speed grid = 2 engines {ik, main} × 4 input sizes {600, 4k, 16k, 30k, by fixture section count} × 2 output
lengths {short = 64 tok, long = 2000 tok}, plus an `ikfa` (ik + `-fa`) variant on the long-context cells.
Quality grid = 2 engines × 4 tasks {needle, summ, reason, code}. (The `30k` cells are reported but excluded
as invalid — §6.1.)

**Self-defense.** A preflight clean-slate gate, a single-instance guard, a memory-headroom guard, and a
cold-cache bias detector each annotate a `flags` column. **This run produced zero flags.**

---

## 4. Results

Cells are matched to the gpt-oss study by *fixture* (section count), but the two models tokenize the same
text differently, so the **actual token depths differ** (Qwen ≈ 10–15 % higher; see the `≈tokens` column).
Comparisons are therefore at *comparable*, not token-identical, depths. All values are medians of 3 reps.

### 4.1 Speed — decode (tok/s)
| cell (≈Qwen tokens) | ik · short | ik · long | main · short | main · long | ikfa · short | ikfa · long |
|---|---|---|---|---|---|---|
| 600 (~0.6k) | **22.49** | **17.13** | 22.06 | 15.17 | — | — |
| 4k (~4.4k) | **17.43** | **12.95** | 11.80 | 8.71 | — | — |
| 16k (~17.9k) | **9.09** | **6.93** | 3.42 | 2.56 | 9.20 | 6.79 |

### 4.2 Speed — prefill (tok/s)
| cell (≈Qwen tokens) | ik · short | ik · long | main · short | main · long | ikfa · short | ikfa · long |
|---|---|---|---|---|---|---|
| 600 | **97.25** | **97.64** | 58.62 | 58.50 | — | — |
| 4k | **65.01** | **63.97** | 40.80 | 40.86 | — | — |
| 16k | **25.66** | **25.44** | 18.94 | 18.90 | 25.16 | 24.91 |

### 4.3 Latency — cold vs warm TTFT (seconds)
| cell | ik cold | ik warm | main cold | main warm | ikfa cold | ikfa warm |
|---|---|---|---|---|---|---|
| 600 (short/long) | 6.16 / 6.53 | ~0.06 | 10.20 / 10.89 | ~0.06 | — | — |
| 4k | 67.34 / 69.05 | ~0.07 | 107.27 / 108.07 | ~0.11 | — | — |
| 16k | 696.59 / 704.16 | ~0.14 | 943.47 / 947.71 | ~0.37 | 710.25 / 719.04 | ~0.14 |

Cold TTFT scales with prompt length and prefill rate; warm TTFT (prefix-cache hit) is ~0.06–0.37s
regardless of depth — i.e. ~**5000×** lower at the 16k cell.

### 4.4 Quality (medians; correctness by deterministic grader)
| task | ≈tokens | ik decode | ik TTFT | ik result | main decode | main TTFT | main result |
|---|---|---|---|---|---|---|---|
| needle (retrieval) | 17,873 | 9.05 | 703.65 | **PASS** (1.0) | 3.44 | 943.10 | **PASS** (1.0) |
| summ (8 figures) | 17,965 | 7.10 | 724.67 | **FAIL** (0.375) | 2.67 | 959.22 | **FAIL** (0.375) |
| reason (arith) | 88 | 18.73 | 1.04 | **PASS** (1.0) | 20.21 | 1.59 | **PASS** (1.0) |
| code (function) | 80 | 18.88 | 0.94 | **PASS** (1.0) | 20.34 | 1.39 | **PASS** (1.0) |

Needle, reason, and code pass on both engines. The summarization task scored **3/8 figures (0.375)**
identically across both engines and all reps — a systematic result discussed in §6.2.

### 4.5 Correctness tally
84 rows, no flags. **72/84 PASS.** The 12 non-PASS rows are fully accounted for:
6 × summ (the §6.2 grader question) and 6 × `spd_main_30k_*` (mainline's correct context-overflow
rejections — §6.1). The ik/ikfa "30k" cells report PASS but are invalid (silent truncation — §6.1).

---

## 5. Comparison with gpt-oss-20b (same box, run `20260614-191322`)

Decode tok/s, **ik_llama.cpp**, medians. Token depths are not identical (gpt-oss tokenizes the same
fixtures lower: its "16k" cell ≈ 15,960 tok vs Qwen's ≈ 17,873 tok), so read these as comparable-depth,
not token-matched.

| cell | gpt-oss · short | gpt-oss · long | Qwen · short | Qwen · long |
|---|---|---|---|---|
| 600 | 13.60 | 13.21 | **22.49** | **17.13** |
| 4k | 12.81 | 12.47 | **17.43** | 12.95 |
| 16k | **10.90** | **10.22** | 9.09 | 6.93 |

Prefill tok/s (ik): gpt-oss 85 / 80 / 51 vs Qwen 97 / 65 / 26 (at 600 / 4k / 16k). Cold TTFT at the 16k
cell (ik): gpt-oss ~308s vs Qwen ~697s.

**Reading.** Qwen decodes faster at short/medium context (≤4k), with a crossover around the 16k cell where
gpt-oss is slightly faster on decode — partly because Qwen's 16k cell is ~12 % deeper in tokens. gpt-oss
**prefills markedly better at depth** (≈2× at 16k), giving it much lower cold TTFT on long prompts. Neither
model can be compared above ~32k on this box (gpt-oss was studied at 32k ctx; Qwen overflowed — §6.1).

---

## 6. Limitations and anomalies

### 6.1 The "30k" cells are invalid (context overflow) — primary caveat
The run used `--ctx-size 32768` (chosen to mirror the gpt-oss study). The "30k" fixture, sized for
gpt-oss's tokenizer, tokenizes to **34,483 tokens** under Qwen's tokenizer — beyond the 32,768 window.
The engines diverged:
- **mainline llama.cpp**: returned `HTTP 400 — "request (34483 tokens) exceeds the available context size
  (32768 tokens)"` (`exceed_context_size_error`). Correct, loud failure → 6 honest FAIL rows.
- **ik_llama.cpp**: did **not** error; it truncated/shifted and reported `prompt_tokens ≈ 18,099` (not
  34,483), producing PASS rows that are **not a real 30k-token decode** (the tell: its "30k" depth ≈ its
  "16k" depth). 

Consequently **all "30k" cells (ik, ikfa, main) are excluded**, and the deepest valid measurement is the
"16k" cell (~17.9k tokens). This is a property of mismatched tokenizers, not of the model's capability:
Qwen3-Coder natively supports 262,144 tokens, and the box has RAM headroom for a 64k window — testing
genuine deep context simply requires a Qwen-tokenizer-calibrated fixture and a larger `--ctx-size`.
**Harness gap noted:** the cold-cache bias detector does not catch context overflow / silent truncation;
an explicit overflow guard should precede any deep-context re-run.

### 6.2 Summarization task scored 0.375 on both engines
The summarization grader checks for 8 specific figures in the model's summary. Qwen reproduced 3/8,
identically across both engines and all 3 reps — i.e. systematic, not noise. Three explanations are
consistent with the data and **were not disambiguated in this run**: (a) this run sampled at temperature
1.0 rather than the intended 0.7 (§2.3, §6.4) — higher temperature plausibly hurts exact figure recall;
(b) the grader's figure-matching was calibrated against gpt-oss output and Qwen formats/represents figures
differently; or (c) Qwen genuinely omits figures from its summary. This affects only the summ quality
verdict, not any speed metric. The other three quality tasks (needle, reason, code) pass on both engines.

### 6.3 Other limitations
- **Single-instance, batch=1** (`--parallel 1`): these are single-user latency/throughput numbers, not
  concurrent-serving or goodput figures.
- **`--ctx-size 32768` only**: 64k/256k contexts were not benchmarked (and the box maxes ~32k for the
  gpt-oss comparison). Cold TTFT at 64k would be substantially larger; not measured.
- **Effective memory bandwidth not directly measured** (only the 76.8 GB/s theoretical peak is stated).
- **Decode "short" vs "long"** differ in output length (64 vs 2000 tokens); the long cells reflect
  sustained decode, the short cells include relatively more fixed overhead.

### 6.4 Sampling override (harness defect, corrected post-run)
The benchmark payload generator hard-codes sampling (`temperature 1.0, top_p 1.0, top_k 0, min_p 0.0`)
into every request body. llama.cpp honors request-body sampling over the server's CLI flags, so the Qwen
runs sampled at those values rather than the intended `0.7 / 0.8 / 20` set on the server; only
`repeat_penalty 1.05` (absent from the body) applied. Verified three ways: the server's `/props` reports
the 0.7 defaults; the saved request bodies contain 1.0; and a greedy probe (`temperature:0, top_k:1` in
the body) returns byte-identical output while no-body-sampling requests vary — proving the body overrides
the server. gpt-oss is unaffected (its intended sampling equals the body's values). Speed metrics are
sampling-independent and unchanged. The generator has since been fixed to defer to the server flags for
future runs.

---

## 7. Reproducibility
- All values from `results/20260618-220001/results.csv` (84 rows) and per-cell raw artifacts under
  `results/20260618-220001/raw/`. Model metadata in §2.2 is read from the loaded GGUF (`server.log`).
- Configuration is single-sourced in `scripts/config.env` (`ACTIVE_MODEL=qwen`, `QWEN_*` vars, `CTX`,
  `THREADS`, engine commits). Re-run: `nohup bash scripts/bench-suite.sh >> results/suite.out 2>&1 &`.
- gpt-oss comparison numbers (§5) are from `results/20260614-191322/results.csv`.

---

## 8. Conclusion
Qwen3-Coder-30B-A3B runs usefully on this CPU-only box: short-context decode (~22 tok/s with ik) exceeds
gpt-oss-20b on the same hardware, and **ik_llama.cpp is the unambiguous engine choice** — faster at every
tested depth, decisively so beyond ~4k. The practical limit is **prefill**: cold first-token latency grows
to ~12–16 minutes at ~18k tokens, so long-prompt workloads depend on prefix caching (warm TTFT ~0.14s) or
a GPU. The headline caveat is methodological: the deepest cells overflowed Qwen's configured 32k window
because the fixtures were tokenizer-calibrated for gpt-oss; a genuine deep-context characterization (at 64k,
Qwen's territory) remains to be run with corrected fixtures and an overflow guard.
