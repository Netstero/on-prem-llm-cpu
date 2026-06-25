#!/usr/bin/env python3
# bench-verify.py — task-specific correctness check for one benchmark response.
# Usage: bench-verify.py <task> <response.json>
# Prints ONE line: "PASS|FAIL <score 0..1> <extracted note>".  Exit code always 0;
# the verdict is in stdout so the harness can record it even on a soft failure.
import json, sys, re, os, importlib.util

here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bp", os.path.join(here, "bench-payloads.py"))
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)


def _text(resp):
    m = resp["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or "")


def v_needle(resp):
    c, r = _text(resp)
    ok = bp.NEEDLE_ANSWER in (c + "\n" + r)
    return ok, 1.0 if ok else 0.0, bp.NEEDLE_ANSWER if ok else "(missing)"


def v_summ(resp):
    c, r = _text(resp)
    txt = c + "\n" + r
    figs = list(bp.SUMM_FACTS.values())
    hit = [f for f in figs if f in txt]
    score = len(hit) / len(figs)
    return score >= 0.75, round(score, 3), f"{len(hit)}/{len(figs)} figures"


def v_reason(resp):
    c, r = _text(resp)
    m = re.findall(r"ANSWER:\s*(-?\d+)", c + "\n" + r)
    ans = m[-1] if m else None
    ok = ans == bp.REASON_ANSWER
    return ok, 1.0 if ok else 0.0, f"ANSWER={ans}"


# Brackets task needs none of these; reject if present (don't exec untrusted constructs).
_UNSAFE = re.compile(r"\b(import|__import__|eval|exec|compile|open|subprocess|os\.|sys\.|"
                     r"globals|locals|getattr|setattr|input)\b")


def v_code(resp):
    c, r = _text(resp)
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", c, re.S) or \
             re.findall(r"```(?:python)?\s*(.*?)```", c + r, re.S)
    if not blocks:
        return False, 0.0, "(no code block)"
    src = blocks[0]
    if _UNSAFE.search(src):
        return False, 0.0, "(unsafe-code-skipped)"
    ns = {}
    try:
        safe = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                for k in ("len", "range", "enumerate", "dict", "list", "set", "str", "bool",
                          "int", "zip", "reversed", "tuple", "sorted", "any", "all")}
        exec(src, {"__builtins__": safe}, ns)
        f = ns.get("is_balanced")
        if not f:
            return False, 0.0, "(no is_balanced)"
        cases = [("()[]{}", True), ("(]", False), ("([)]", False), ("", True), ("{[()]}", True),
                 ("(((", False), ("a(b)c[d]", True), ("}{", False)]
        passed = sum(1 for s, exp in cases if bool(f(s)) == exp)
        score = passed / len(cases)
        return score == 1.0, round(score, 3), f"{passed}/{len(cases)} cases"
    except Exception as e:
        return False, 0.0, f"(exec error: {type(e).__name__})"


V = {"needle_short": v_needle, "needle_long": v_needle, "needle": v_needle,
     "summ": v_summ, "reason": v_reason, "code": v_code}

if __name__ == "__main__":
    task, path = sys.argv[1], sys.argv[2]
    try:
        ok, score, extra = V[task](json.load(open(path)))
        print(f"{'PASS' if ok else 'FAIL'} {score} {extra}")
    except Exception as e:
        print(f"FAIL 0.0 (verify-error: {type(e).__name__}: {e})")
