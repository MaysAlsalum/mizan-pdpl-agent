"""Error analysis for one results file: every missed violation and every false alarm, with the reason.
No model calls. Use it on dev only, never to tune on test.

Usage:
    python -m eval.analyze                     # latest agent run on dev
    python -m eval.analyze eval/results/<file>.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent


def latest(prefix: str = "agent_dev_") -> Path:
    files = sorted((ROOT / "eval" / "results").glob(f"{prefix}*.json"))
    if not files:
        raise SystemExit(f"no results matching {prefix}*")
    return files[-1]


def best_clause(text: str, clauses: list[dict]) -> dict | None:
    if not clauses:
        return None
    score, c = max(((fuzz.partial_ratio(text, c["text"]), c) for c in clauses), key=lambda x: x[0])
    return c if score >= 80 else None


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest()
    run = json.loads(path.read_text(encoding="utf-8"))
    print(f"{run['system']} on {run['split']}  ({path.name})\n")
    causes: Counter = Counter()

    for item in run["per_doc"]:
        doc = json.loads((ROOT / "data" / run["split"] / f"{item['doc_id']}.json").read_text(encoding="utf-8"))
        out = item["output"]
        if out.get("error"):
            print(f"[{doc['doc_id']}] CRASHED: {out['error'][:120]}")
            causes["crash"] += 1
            continue
        found = [f for f in out.get("findings", [])]
        seg = out.get("clauses", [])
        assess = out.get("assessments", [])

        for g in (c for c in doc["clauses"] if c["label"] == "violation"):
            hit = any(f["rule_id"] == g["rule_id"] and fuzz.partial_ratio(f["evidence"], g["text"]) >= 80 for f in found)
            if hit:
                continue
            k = best_clause(g["text"], seg)
            if seg and k is None:
                why = "segmentation: clause not found as one piece"
            elif seg and g["rule_id"] not in k["rules"]:
                why = "rule selection: rule never sent with this clause" if k["rules"] else "gate: clause judged not about personal data"
            elif any(e["rule_id"] == g["rule_id"] and fuzz.partial_ratio(e.get("evidence", ""), g["text"]) >= 80
                     for e in out.get("escalations", [])):
                why = "escalated instead of flagged"
            else:
                a = next((a for a in assess if k and a.get("clause_id") == k["id"] and a.get("rule_id") == g["rule_id"]), None)
                why = (f"judge said {a.get('verdict')} ({a.get('confidence')}): {str(a.get('reason', ''))[:90]}"
                       if a else "judge gave no verdict for this pair" if seg else "missed")
            causes[why.split(":")[0].split(" (")[0]] += 1
            tag = "subtle" if g.get("subtle") else "clear"
            print(f"[{doc['doc_id']}] MISSED {g['rule_id']} ({tag}) -> {why}\n    {g['text'][:170]}\n")

        for f in found:
            match = [c for c in doc["clauses"] if fuzz.partial_ratio(f["evidence"], c["text"]) >= 80]
            if any(c["label"] == "violation" and c["rule_id"] == f["rule_id"] for c in match):
                continue
            if any(c["label"] in ("violation", "ambiguous") for c in match):
                continue  # unverifiable second rule or borderline: not scored as an error
            label = match[0]["label"] if match else "no matching clause"
            causes[f"false alarm on {label}"] += 1
            print(f"[{doc['doc_id']}] FALSE ALARM {f['rule_id']} on {label} clause (conf {f.get('confidence')})\n"
                  f"    {f['evidence'][:170]}\n")

    print("Summary of causes:")
    for why, n in causes.most_common():
        print(f"  {n:3}  {why}")


if __name__ == "__main__":
    main()