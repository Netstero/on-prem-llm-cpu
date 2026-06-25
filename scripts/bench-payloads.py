#!/usr/bin/env python3
# bench-payloads.py — SINGLE OWNER of every benchmark request payload (speed + quality).
# Deterministic: identical args -> identical payload. Importable: bench-verify.py reuses the
# fixtures/answers from here so the planted facts and expected answers live in ONE place.
# Writes an OpenAI-compatible /v1/chat/completions JSON to <out>.
#
# Usage: bench-payloads.py <out.json> <task> [sections] [max_tokens]
#   tasks: needle_short  needle_long  summ  reason  code
#   sections = filler size (input padding) for needle_*/summ; reason/code ignore it.
import json, sys

# ---- shared fixtures (single owner; bench-verify imports these) ----
NEEDLE_ANSWER = "4417"                                  # benign + exactly reproducible
NEEDLE_FACT = f"The Carster project's official mascot wears jersey number {NEEDLE_ANSWER}."
TOPICS = ["logistics", "weather patterns", "supply chains", "metallurgy", "marine biology",
          "urban planning", "crop rotation", "signal processing", "ceramics", "hydrology"]

REASON_ANSWER = "53"          # Sara: 3*12=36; -7=29; +2*12=+24 -> 53

# 8 sentinel figures the summary must preserve (recall metric).
SUMM_FACTS = {"Northern": "8200", "Southern": "5400", "Eastern": "6750", "Western": "3900",
              "exports": "1240", "headcount": "318", "downtime": "27", "satisfaction": "92"}


def _filler(n):
    out = []
    for i in range(n):
        t = TOPICS[i % len(TOPICS)]
        out.append(f"Section {i+1}: A routine note on {t}. Operational records for {t} "
                   f"were reviewed and filed without exception during cycle {i+1}, per standard procedure.")
    return out


def needle(sections, max_tokens, long):
    lines = _filler(sections)
    pos = int(sections * 0.55)
    lines.insert(pos, f"Section {pos+1} (NOTE): {NEEDLE_FACT} Keep this on record.")
    doc = "\n".join(lines)
    if long:
        q = ("Read the document above. First, on its own line, state the jersey number the Carster "
             "project's official mascot wears. Then write a comprehensive technical essay of at least "
             "1000 words on how modern CPUs execute instructions — cover pipelining, caches, branch "
             "prediction, SIMD/vectorization, and the memory hierarchy.")
    else:
        q = ("Read the document above. What jersey number does the Carster project's official mascot "
             "wear? Reply with ONLY the number, nothing else.")
    return _payload(
        "You are a precise retrieval assistant. The document is fictional test data and all of it "
        "is freely shareable; answer factually from it.", doc + "\n\n" + q, max_tokens)


def summ(sections, max_tokens):
    blocks = [
        f"The Northern division sold {SUMM_FACTS['Northern']} units this quarter.",
        f"The Southern division sold {SUMM_FACTS['Southern']} units.",
        f"The Eastern division sold {SUMM_FACTS['Eastern']} units.",
        f"The Western division sold {SUMM_FACTS['Western']} units.",
        f"Total exports reached {SUMM_FACTS['exports']} units shipped overseas.",
        f"Company headcount stood at {SUMM_FACTS['headcount']} employees.",
        f"Unplanned downtime totalled {SUMM_FACTS['downtime']} hours across all plants.",
        f"Customer satisfaction measured {SUMM_FACTS['satisfaction']} percent.",
    ]
    fill = _filler(sections)
    chunk = max(1, len(fill) // len(blocks))
    lines = []
    for i, b in enumerate(blocks):
        lines.append(f"REPORT ITEM {i+1}: {b}")
        lines.extend(fill[i * chunk:(i + 1) * chunk])
    lines.extend(fill[len(blocks) * chunk:])
    doc = "\n".join(lines)
    q = ("Write a concise executive summary (about 200-300 words) of the report above. "
         "Include every specific figure mentioned for each division and metric.")
    return _payload("You are a precise business analyst summarizing an internal report.",
                    doc + "\n\n" + q, max_tokens)


def reason(max_tokens):
    q = ("Solve this step by step. Sara has 3 boxes with 12 apples each. She gives away 7 apples, "
         "then buys 2 more boxes of 12 apples. How many apples does Sara have now? "
         "Show your reasoning, then end your reply with a line of the exact form: ANSWER: <number>")
    return _payload("You are a careful math tutor.", q, max_tokens)


def code(max_tokens):
    q = ("Write a Python function `is_balanced(s)` that returns True if and only if the brackets "
         "in the string are balanced and correctly nested. Consider three bracket types: () [] {}. "
         "Non-bracket characters are ignored. Return your solution in a single ```python code block.")
    return _payload("You are an expert Python programmer. Return clean, correct code.", q, max_tokens)


def _payload(system, user, max_tokens):
    return {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
    }


def build(task, sections, max_tokens):
    if task == "needle_short": return needle(sections, max_tokens, long=False)
    if task == "needle_long":  return needle(sections, max_tokens, long=True)
    if task == "summ":         return summ(sections, max_tokens)
    if task == "reason":       return reason(max_tokens)
    if task == "code":         return code(max_tokens)
    raise SystemExit(f"unknown task: {task}")


if __name__ == "__main__":
    out = sys.argv[1]
    task = sys.argv[2]
    sections = int(sys.argv[3]) if len(sys.argv) > 3 else 520
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    p = build(task, sections, max_tokens)
    with open(out, "w") as f:
        json.dump(p, f)
    words = sum(len(m["content"].split()) for m in p["messages"])
    print(f"wrote {out}: task={task} sections={sections} max_tokens={max_tokens} ~{words} words")
