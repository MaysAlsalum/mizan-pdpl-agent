"""Baseline: one prompt with the whole document and the rule list. No splitting, no retrieval, no checks."""
from __future__ import annotations

import time

from src.catalog import load_rules
from src.llm import call

PROMPT = """You are a data protection compliance reviewer for Saudi Arabia's Personal Data Protection Law.
Review the document below against these rules and list every clause that violates one of them.

Rules:
{rules}

For each violation return the rule id, a confidence from 0 to 1, and the evidence: an exact quote
copied from the document. Do not report clauses that comply.

Return JSON only: {{"findings": [{{"rule_id": "R01", "confidence": 0.9, "evidence": "exact quote"}}]}}

Document:
{document}"""


def run(doc: dict, role: str = "small", include_hidden: bool = False) -> dict:
    rules = "\n".join(f"- {r['id']} {r['title_en']}: {r['requirement']}" for r in load_rules(include_hidden))
    start = time.perf_counter()
    res = call(PROMPT.format(rules=rules, document=doc["full_text"]), role=role, json_mode=True)
    raw = (res.data or {}).get("findings", []) if isinstance(res.data, dict) else []
    findings = [{"rule_id": f.get("rule_id"), "verdict": "violation",
                 "confidence": f.get("confidence"), "evidence": f.get("evidence", ""), "fix": None}
                for f in raw if isinstance(f, dict)]
    return {"doc_id": doc["doc_id"], "findings": findings, "escalations": [],
            "cost_usd": res.cost_usd, "latency_s": round(time.perf_counter() - start, 2), "llm_calls": 1}