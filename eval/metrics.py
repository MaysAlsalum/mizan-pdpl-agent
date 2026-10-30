"""Score system outputs against the planted labels.

A finding is correct only if its rule_id matches AND its evidence points at the planted clause
(fuzzy match, so small quoting differences and noisy text still count). Guessing the right rule
from the wrong clause scores as a false positive.
"""
from __future__ import annotations

from collections import defaultdict

from rapidfuzz import fuzz

EVIDENCE_MATCH = 80   # evidence vs clause text
GROUNDED = 90         # evidence vs full document text


def _matches(evidence: str, text: str, threshold: int = EVIDENCE_MATCH) -> bool:
    return bool(evidence) and bool(text) and fuzz.partial_ratio(evidence, text) >= threshold


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def score_doc(doc: dict, out: dict) -> dict:
    """Counts for one document. `out` is a system output in the shared format."""
    clauses = doc["clauses"]
    gold = [c for c in clauses if c["label"] == "violation"]
    traps = [c for c in clauses if c["label"] == "trap"]
    ambiguous = [c for c in clauses if c["label"] == "ambiguous"]
    findings = [f for f in out.get("findings", []) if f.get("verdict", "violation") == "violation"]

    matched_gold: set[str] = set()
    per_rule = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    tp = fp = trap_hits = grounded = unverified = 0
    for f in findings:
        ev, rid = f.get("evidence", ""), f.get("rule_id")
        grounded += _matches(ev, doc["full_text"], GROUNDED)
        hit = next((g for g in gold if g["clause_id"] not in matched_gold
                    and g["rule_id"] == rid and _matches(ev, g["text"])), None)
        if hit:
            matched_gold.add(hit["clause_id"])
            tp += 1
            per_rule[rid]["tp"] += 1
            continue
        if any(_matches(ev, a["text"]) for a in ambiguous):
            continue  # flagging a genuinely borderline clause is neither right nor wrong
        if any(_matches(ev, g["text"]) for g in gold):
            # A planted violating clause can also break a second rule we never labelled
            # (or this repeats a finding already counted). We cannot verify it, so it is
            # neither credited nor penalised; recall stays strict on the labelled rule.
            unverified += 1
            continue
        fp += 1
        per_rule[rid]["fp"] += 1
        trap_hits += any(_matches(ev, t["text"]) for t in traps)

    for g in gold:
        if g["clause_id"] not in matched_gold:
            per_rule[g["rule_id"]]["fn"] += 1
    fn = len(gold) - len(matched_gold)
    subtle = [g for g in gold if g.get("subtle")]
    subtle_found = sum(g["clause_id"] in matched_gold for g in subtle)

    escalations = out.get("escalations", [])
    esc_on_ambiguous = sum(any(_matches(e.get("evidence", ""), a["text"]) for e in escalations)
                           for a in ambiguous)

    return {"tp": tp, "fp": fp, "fn": fn, "per_rule": dict(per_rule),
            "traps": len(traps), "trap_hits": trap_hits,
            "subtle": len(subtle), "subtle_found": subtle_found, "unverified": unverified,
            "findings": len(findings), "grounded": grounded,
            "ambiguous": len(ambiguous), "ambiguous_escalated": esc_on_ambiguous,
            "escalations": len(escalations),
            "cost_usd": out.get("cost_usd", 0.0), "latency_s": out.get("latency_s", 0.0),
            "llm_calls": out.get("llm_calls", 0)}


def aggregate(scores: list[dict]) -> dict:
    """Micro-averaged metrics over a split, plus per-rule F1."""
    if not scores:
        return {}
    s = lambda k: sum(x[k] for x in scores)  # noqa: E731
    rules = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for x in scores:
        for rid, c in x["per_rule"].items():
            for k in c:
                rules[rid][k] += c[k]
    n = len(scores)
    return {
        **_prf(s("tp"), s("fp"), s("fn")),
        "subtle_recall": round(s("subtle_found") / s("subtle"), 3) if s("subtle") else None,
        "trap_fp_rate": round(s("trap_hits") / s("traps"), 3) if s("traps") else None,
        "grounding_rate": round(s("grounded") / s("findings"), 3) if s("findings") else None,
        "ambiguous_escalation_rate": round(s("ambiguous_escalated") / s("ambiguous"), 3) if s("ambiguous") else None,
        "escalations_per_doc": round(s("escalations") / n, 2),
        "cost_per_doc_usd": round(s("cost_usd") / n, 5),
        "latency_per_doc_s": round(s("latency_s") / n, 2),
        "llm_calls_per_doc": round(s("llm_calls") / n, 1),
        "docs": n,
        "per_rule_f1": {rid: _prf(**c)["f1"] for rid, c in sorted(rules.items())},
    }