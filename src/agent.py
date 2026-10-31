"""The compliance agent: segment -> pick rules per clause -> judge with evidence -> recheck or escalate -> fix and re-verify.

Every stage can be switched off for ablations via `ablate`:
    no_segment   judge the whole document as one block
    no_retrieval send every rule with every clause
    no_traps     drop the trap examples from the judge prompt
    no_grounding accept the model's evidence quote without checking it
    with_recheck add a second review of borderline verdicts (off by default since v3:
                 with the same model as the judge it added false alarms, see results_log)
    doc_type_filter only send rules listed for the guessed document type (off since v3:
                 a wrong guess hid whole rules)
    no_fix       skip the fix-and-verify loop
"""
from __future__ import annotations

import re
import time

from rapidfuzz import fuzz

from src.catalog import load_rules
from src.llm import CONFIG, LLMResult, call

TH = CONFIG["thresholds"]
HIGH = TH["recheck_band"][1]
LOW = TH["escalate_below"]
MAX_FIX = TH["max_fix_attempts"]
CLAUSES_PER_CALL = 6
AR_TO_EN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# ---------- 1. segmentation (no model) ----------

MARKER = re.compile(
    r"^[ \t]*(?:"
    r"(?P<a>[0-9٠-٩]+)\s*[-.)]"                     # 1.  ١-  3)
    r"|\((?P<b>[0-9٠-٩]+)\)"                         # (1)
    r"|(?:Article|Section|Clause)\s+(?P<c>\d+)\s*[:.]?"  # Article 1:
    r"|المادة\s*\(?(?P<d>[0-9٠-٩]+)\)?\s*:?"          # المادة (١):
    r")\s*", re.MULTILINE)
PAGE_MARK = re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE)


def segment(text: str) -> list[dict]:
    """Split on clause numbers that run in sequence (1, 2, 3...), so a stray '(60)' at a line start is ignored."""
    text = PAGE_MARK.sub(" ", text)
    cuts, expected = [], 1
    for m in MARKER.finditer(text):
        num = int(next(g for g in m.group("a", "b", "c", "d") if g).translate(AR_TO_EN))
        if num == expected:
            cuts.append(m)
            expected += 1
    clauses = []
    if len(cuts) >= 2:
        for i, m in enumerate(cuts):
            end = cuts[i + 1].start() if i + 1 < len(cuts) else len(text)
            clauses.append(text[m.end():end])
    else:  # no numbering: fall back to paragraphs
        clauses = [p for p in re.split(r"\n\s*\n", text) if p.strip()][1:] or [text]
    out = []
    for i, c in enumerate(clauses, 1):
        clean = re.sub(r"\s+", " ", c).strip()
        if clean:
            out.append({"id": f"k{i}", "text": clean})
    return out


# ---------- 2. rule selection (no model) ----------

DOC_TYPE_HINTS = {
    "vendor_contract": ["vendor", "supplier", "service provider", "agreement", "المورد", "مزود الخدمة", "اتفاقية"],
    "employment_form": ["applicant", "candidate", "employment", "job", "المتقدم", "التوظيف", "وظيفة"],
    "privacy_policy": ["privacy policy", "سياسة الخصوصية", "this policy", "هذه السياسة"],
    "mobile_app_terms": ["app", "application", "download", "التطبيق", "تحميل"],
}


def guess_doc_type(text: str) -> str:
    low = text[:600].lower()
    scores = {t: sum(low.count(h.lower()) for h in hints) for t, hints in DOC_TYPE_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "unknown"


# v3: the gate also knows the personal-data FIELDS people write instead of the word "data"
GENERIC_DATA_WORDS = [
    "data", "information", "personal", "record", "file", "details", "applicant", "user", "customer",
    "client", "employee", "candidate", "profile", "account", "identity", "national id", "id card", "name",
    "email", "e-mail", "phone", "address", "date of birth", "marital", "family", "location", "health",
    "بيانات", "معلومات", "شخصي", "سجل", "ملف", "تفاصيل", "المستخدم", "العميل", "الموظف", "المتقدم",
    "هوية", "الهوية", "الاسم", "البريد", "الجوال", "الهاتف", "العنوان", "تاريخ الميلاد", "الحالة الاجتماعية",
    "أفراد الأسرة", "الأسرة", "جهة العمل", "مكان العمل", "الموقع", "الصحية", "التسجيل", "حساب",
]


def candidate_rules(clause: str, rules: list[dict], doc_type: str, use_retrieval: bool,
                    use_doc_type: bool = False) -> list[dict]:
    """Gate: a clause that never touches personal data goes to no rule at all; one that does is
    checked against every rule (ranked by keyword overlap).
    v1 kept only the top-3 keyword matches: subtle wording slipped past it.
    v2 kept only the rules for the guessed document type: a wrong guess hid whole rules."""
    if not use_retrieval:
        return rules
    low = clause.lower()
    words = GENERIC_DATA_WORDS + [k for r in rules for k in r["keywords_ar"] + r["keywords_en"]]
    if not any(w.lower() in low for w in words):
        return []
    pool = [r for r in rules if doc_type in r["applies_to"]] if use_doc_type and doc_type != "unknown" else rules
    score = lambda r: sum(k.lower() in low for k in r["keywords_ar"] + r["keywords_en"])  # noqa: E731
    return sorted(pool, key=score, reverse=True)


# ---------- 3. judging ----------

def _rule_block(r: dict, with_traps: bool) -> str:
    lines = [f"  {r['id']} {r['title_en']} ({r['source']}): {r['requirement']}",
             f"    Violation looks like: {r['violation_looks_like']}"]
    if with_traps:
        lines.append(f"    Looks suspicious but COMPLIES (do not flag things like this): {r['trap_example']}")
    return "\n".join(lines)


JUDGE_PROMPT = """You are a careful data protection compliance reviewer (Saudi Personal Data Protection Law).
For every clause, assess it ONLY against the rules listed under it. The clauses come from this
document, so judge each one in its context (what is collected, from whom, and for what):
Document: {title}

verdict: "violation" if the clause breaks the rule, "compliant" if it respects it or does not really
concern it, "unclear" if a careful lawyer could not decide from the text.
confidence: 0 to 1, how sure you are of the verdict.
evidence: the shortest exact quote from the clause that supports the verdict (copied character for character).

{blocks}

Return JSON only, one entry per clause-rule pair listed above:
{{"assessments": [{{"clause_id": "k1", "rule_id": "R01", "verdict": "compliant", "confidence": 0.9, "evidence": "...", "reason": "one short sentence"}}]}}"""

RECHECK_PROMPT = """A first reviewer was unsure about this clause. Review it independently and carefully.
Rule {rid} {title} ({source}): {requirement}
A violation looks like: {looks}
{trap}
Clause: {clause}

Is this clause a violation of the rule? Return JSON only:
{{"verdict": "violation" | "compliant" | "unclear", "confidence": 0.0, "evidence": "exact quote", "reason": "one short sentence"}}"""

FIX_PROMPT = """Rewrite this clause so it complies with the rule, keeping its business intent, tone and language
({lang}). Change as little as possible.
Rule {rid}: {requirement}
Problem: {reason}
{feedback}
Clause: {clause}

Return JSON only: {{"fixed_clause": "..."}}"""


class _Meter:
    """Per-document cost and call counter."""

    def __init__(self) -> None:
        self.cost, self.calls = 0.0, 0

    def __call__(self, res: LLMResult) -> LLMResult:
        self.cost += res.cost_usd
        self.calls += 1
        return res


def _as_list(data, key: str) -> list:
    if isinstance(data, dict):
        data = data.get(key, [])
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def _grounded(quote: str, clause: str) -> bool:
    return bool(quote) and fuzz.partial_ratio(quote, clause) >= 90


def run(doc: dict, include_hidden: bool = False, ablate: str | None = None) -> dict:
    flags = set((ablate or "").split("+")) - {""}
    rules = load_rules(include_hidden)
    by_id = {r["id"]: r for r in rules}
    text = doc["full_text"]
    meter = _Meter()
    start = time.perf_counter()
    trace = []

    # 1-2. segment and pick rules
    clauses = [{"id": "k1", "text": re.sub(r"\s+", " ", text)}] if "no_segment" in flags else segment(text)
    doc_type = guess_doc_type(text)
    title = text.strip().splitlines()[0][:200] if text.strip() else ""
    pairs = {c["id"]: candidate_rules(c["text"], rules, doc_type, "no_retrieval" not in flags,
                                      "doc_type_filter" in flags) for c in clauses}
    todo = [c for c in clauses if pairs[c["id"]]]
    trace.append(f"doc_type={doc_type} clauses={len(clauses)} with_candidates={len(todo)}")

    # 3. judge in batches
    assessments = []
    for i in range(0, len(todo), CLAUSES_PER_CALL):
        batch = todo[i:i + CLAUSES_PER_CALL]
        blocks = "\n\n".join(
            f"Clause {c['id']}: {c['text']}\nRules to check:\n"
            + "\n".join(_rule_block(r, "no_traps" not in flags) for r in pairs[c["id"]])
            for c in batch)
        res = meter(call(JUDGE_PROMPT.format(title=title, blocks=blocks), role="small", json_mode=True))
        assessments += _as_list(res.data, "assessments")

    text_of = {c["id"]: c["text"] for c in clauses}
    findings, escalations, seen = [], [], set()
    for a in assessments:
        cid, rid = a.get("clause_id"), a.get("rule_id")
        verdict = str(a.get("verdict", "")).lower()
        if cid not in text_of or rid not in by_id or verdict == "compliant" or (cid, rid) in seen:
            continue
        seen.add((cid, rid))
        clause, rule = text_of[cid], by_id[rid]
        try:
            conf = float(a.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        evidence, reason = str(a.get("evidence", "")), str(a.get("reason", ""))

        # grounding: the quote must exist in the clause
        if "no_grounding" not in flags and not _grounded(evidence, clause):
            if len(clause) <= 1500:
                evidence, conf = clause, conf - 0.1   # anchor to the clause we segmented
            else:
                escalations.append({"rule_id": rid, "evidence": evidence, "reason": "evidence not found in text"})
                continue

        # 4. confident -> accept; borderline -> recheck with the larger model; weak -> escalate
        if "with_recheck" in flags and not (verdict == "violation" and conf >= HIGH):
            if conf < LOW and verdict != "unclear":
                escalations.append({"rule_id": rid, "evidence": evidence, "reason": f"low confidence ({conf:.2f})"})
                continue
            trap = (f"Looks suspicious but COMPLIES: {rule['trap_example']}" if "no_traps" not in flags else "")
            rc = meter(call(RECHECK_PROMPT.format(
                rid=rid, title=rule["title_en"], source=rule["source"], requirement=rule["requirement"],
                looks=rule["violation_looks_like"], trap=trap, clause=clause), role="large", json_mode=True))
            r2 = rc.data if isinstance(rc.data, dict) else {}
            v2 = str(r2.get("verdict", "")).lower()
            try:
                c2 = float(r2.get("confidence", 0.5))
            except (TypeError, ValueError):
                c2 = 0.5
            if v2 == "compliant":
                trace.append(f"{cid}/{rid}: recheck cleared it")
                continue
            if not (v2 == "violation" and c2 >= HIGH):
                escalations.append({"rule_id": rid, "evidence": evidence,
                                    "reason": f"reviewers disagree or unsure ({verdict} {conf:.2f} / {v2} {c2:.2f})"})
                continue
            conf, reason = c2, r2.get("reason", reason)
        elif verdict != "violation" or conf < LOW:
            reason_esc = "judged unclear" if verdict != "violation" else f"low confidence ({conf:.2f})"
            escalations.append({"rule_id": rid, "evidence": evidence, "reason": reason_esc})
            continue

        finding = {"rule_id": rid, "verdict": "violation", "confidence": round(conf, 2),
                   "evidence": evidence, "reason": reason, "source": rule["source"],
                   "clause": clause, "fix": None, "fix_verified": None}

        # 5. fix and re-verify
        if "no_fix" not in flags:
            feedback = ""
            for attempt in range(1, MAX_FIX + 1):
                fx = meter(call(FIX_PROMPT.format(
                    lang="Arabic" if doc.get("language") == "ar" else "English", rid=rid,
                    requirement=rule["requirement"], reason=reason, feedback=feedback, clause=clause),
                    role="large", json_mode=True))
                fixed = (fx.data or {}).get("fixed_clause", "") if isinstance(fx.data, dict) else ""
                if not fixed:
                    break
                chk = meter(call(JUDGE_PROMPT.format(title=title,
                    blocks=f"Clause k1: {fixed}\nRules to check:\n{_rule_block(rule, True)}"),
                    role="small", json_mode=True))
                verdicts = [str(x.get("verdict", "")).lower() for x in _as_list(chk.data, "assessments")]
                finding["fix"] = fixed
                if verdicts and verdicts[0] == "compliant":
                    finding["fix_verified"] = True
                    break
                finding["fix_verified"] = False
                feedback = "Your previous rewrite still did not comply. Make the obligation explicit."
            if finding["fix"] and not finding["fix_verified"]:
                escalations.append({"rule_id": rid, "evidence": evidence,
                                    "reason": f"fix failed after {MAX_FIX} attempts", "draft_fix": finding["fix"]})
        findings.append(finding)

    return {"doc_id": doc["doc_id"], "doc_type_guess": doc_type, "findings": findings,
            "escalations": escalations, "trace": trace, "assessments": assessments,
            "clauses": [{"id": c["id"], "text": c["text"], "rules": [r["id"] for r in pairs[c["id"]]]} for c in clauses],
            "cost_usd": round(meter.cost, 6),
            "latency_s": round(time.perf_counter() - start, 2), "llm_calls": meter.calls}