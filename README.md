# on-prem-llm-cpu

**How cheaply can you run a capable LLM on-premises, CPU-only — with receipts.**

A reproducible benchmark of two Mixture-of-Experts models — OpenAI's **gpt-oss-20b** and
**Qwen3-Coder-30B-A3B** — served via `llama.cpp` on a single ~2016-era Xeon workstation with **no GPU**.
Two inference engines, four input sizes, two output regimes, four auto-graded quality tasks, three
repetitions each — **168 measured runs**, every figure traceable to a committed CSV.

> No cloud. No GPU. One used workstation. Real numbers, honest caveats.

License: Apache-2.0

---

## TL;DR — results at a glance

All figures are **medians of 3 reps**, `ik_llama.cpp` engine (the better of the two for both models).

| Model | total / active params | decode @ ~0.6k ctx | decode @ ~16k ctx | cold TTFT @ ~16k | warm TTFT |
|---|---|---|---|---|---|
| **gpt-oss-20b** (MXFP4) | 20.9B / 3.6B | **13.6 tok/s** | 10.9 tok/s | ~308 s (≈5 min) | ~0.10 s |
| **Qwen3-Coder-30B-A3B** (Q4) | 30.5B / 3.3B | **22.5 tok/s** | 9.1 tok/s | ~697 s (≈12 min) | ~0.14 s |

**The shape of the result:** decode (token generation) is usable; **cold prefill is the wall** — a fresh
large prompt takes *minutes* to first token on CPU. But that cost is **one-time per context**: the same
prompt against a warm prefix cache returns the first token in ~0.1 s. So the honest verdict isn't
"usable / not usable" — it's an **envelope**: genuinely useful and cheap for modest-context or
cache-reused workloads (chat, RAG over stable corpora); unsuited to cold, large, ad-hoc prompts — the
case that justifies a GPU.

---

## Key findings

1. **A ~2016 used CPU workstation runs 20B–30B MoE models correctly**, CPU-only — retrieval to ~16–30k
   tokens, reasoning, and executable code.
2. **Decode is usable (~9–22 tok/s depending on model and depth); cold prefill is the bottleneck**
   (minutes to first token at depth), but one-time per context (warm ≈0.1 s).
3. **`ik_llama.cpp` beats mainline `llama.cpp` at depth** for both models — decisively so. For gpt-oss,
   mainline wins only short-prompt chat decode; for Qwen, ik wins or ties everywhere. It's the default.
4. **The two models trade places by context length:** Qwen decodes faster at short/medium context;
   gpt-oss prefills better at depth (lower cold latency). The crossover is around ~16k tokens.
5. **Flash-attention (`-fa`) makes no measurable difference** on CPU at these depths — a clean negative
   result, confirmed for both models.

Full analysis, with every table and caveat:
- 📄 **[Combined case study](docs/CASE-STUDY.md)** — both models + head-to-head (start here).
- 📄 [gpt-oss-20b study](docs/gpt-oss-20b.md) · [Qwen3-Coder-30B study](docs/qwen3-coder-30b.md)
- 🔬 [Methodology](docs/METHODOLOGY.md) — what each test is, why, and what's measured.
- 📓 [Lab notebook](docs/LAB-NOTEBOOK.md) — the honest, append-only build log (the messy reality).
- 📊 [Raw data](results/) — the `results.csv` files every number is computed from.

---

## The hardware

| | |
|---|---|
| Machine | Dell Precision T5810 workstation (~2016-era, used-market class) |
| CPU | Intel Xeon E5-2680 v4 — 14 cores / 28 threads, Broadwell, **AVX2, no AVX-512** |
| Memory | 64 GB DDR4-2400, quad-channel (theoretical ≈76.8 GB/s) |
| GPU | none — **CPU-only** |
| Host | unprivileged Proxmox LXC, Debian; 11 inference threads |

CPU inference here is **memory-bandwidth-bound** for decode and **compute-bound** for prefill — which is
exactly why the numbers look the way they do.

---

## Reproduce it

The scripts run **on the target CPU machine** (a Linux box or LXC). They build both engines from source,
fetch the model, serve it via `systemd`, and run the benchmark grid.

```bash
# 1. configure (copy the example, fill in your model/threads/paths)
cp scripts/config.env.example scripts/config.env
$EDITOR scripts/config.env

# 2. install: deps → build llama.cpp + ik_llama.cpp → fetch model → serve via systemd
bash scripts/install.sh

# 3. benchmark (detached; ~9–11 h for the full grid, resumable by RUN_ID)
nohup bash scripts/bench-suite.sh >> results/suite.out 2>&1 &

# 4. summarize a run into median tables
python3 scripts/bench-report.py results/<run-id>/results.csv
```

The harness is **self-defending**: a pre-flight clean-slate gate, a single-instance guard, a
memory-headroom check, and a cold-cache bias detector — anomalies are flagged in the CSV and summarized
in a data-health report, so an unattended overnight run is trustworthy. See
[METHODOLOGY.md](docs/METHODOLOGY.md).

> The author's own SSH wrapper and secrets are intentionally **not** included; copy `config.env.example`
> to `config.env` (gitignored) and run the scripts directly on your box.

---

## Honest caveats (the short version)

- **Single-stream only** (`--parallel 1`): per-request latency / single-user throughput — **not**
  concurrent throughput, goodput, or p99 tails.
- **Qwen's deepest cells are excluded as invalid:** the gpt-oss-calibrated fixtures tokenize to 34,483
  tokens under Qwen's tokenizer, overflowing the 32k context window used for the run. Mainline rejected
  it correctly; ik silently truncated. Deepest *valid* Qwen depth is ~18k tokens. (Details in the case
  study §5.6 — it's a useful lesson in cross-tokenizer benchmarking.)
- **3 reps, median** (no confidence intervals / p99). Variance was near-zero.
- **Effective memory bandwidth is not directly measured** — only the theoretical peak is stated.

---

## How this was built

The install, the benchmark harness, the runs, and these writeups were produced in collaboration with
**[Claude Code](https://claude.com/claude-code)** (Anthropic) — driven, reviewed, and verified by a human.
The [lab notebook](docs/LAB-NOTEBOOK.md) is the unedited, append-only record of that process, including the
wrong turns and the bug that produced one invalid cell. Every published number was cross-checked against
the raw CSVs before release.

## License
[Apache-2.0](LICENSE). The benchmarked models are licensed by their respective authors (both Apache-2.0).
