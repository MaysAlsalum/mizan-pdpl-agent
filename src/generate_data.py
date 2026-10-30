"""Generate synthetic documents with planted violations, traps and ambiguous clauses.

Python decides the plan (document type, language, which rules, which labels); the model only writes
the text. Labels are therefore known by construction. A second model call checks every planted clause,
and clauses that do not read as intended are regenerated or dropped (and logged).

Usage:
    python -m src.generate_data --split dev --n 30 --seed 42
Splits: dev, test, robust_newtype, robust_noisy, robust_newrule
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.catalog import load_rules
from src.llm import CONFIG, TOTALS, call

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

MAIN_TYPES = ["privacy_policy", "vendor_contract", "employment_form"]
DOC_TYPE_DESC = {
    "privacy_policy": "a privacy policy for a Saudi online service",
    "vendor_contract": "a services and data processing agreement between a Saudi company and a technology vendor",
    "employment_form": "a job application form of a Saudi company, with a data processing declaration",
    "mobile_app_terms": "the terms of use of a Saudi mobile app",
}
SPLITS = {
    "dev": {"doc_types": MAIN_TYPES},
    "test": {"doc_types": MAIN_TYPES},
    "robust_newtype": {"doc_types": ["mobile_app_terms"]},
    "robust_noisy": {"doc_types": MAIN_TYPES, "noisy": True},
    "robust_newrule": {"doc_types": ["vendor_contract", "privacy_policy"], "include_hidden": True,
                       "force_rule": "R08"},
}
EXPECTED_VERDICT = {"violation": "violation", "trap": "compliant", "ambiguous": "unclear"}
MAX_ATTEMPTS = 2
AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


# ---------- plan (pure Python, seeded) ----------

def plan_doc(rng: random.Random, split: str, doc_id: str) -> dict:
    cfg = SPLITS[split]
    doc_type = rng.choice(cfg["doc_types"])
    language = "ar" if rng.random() < 0.6 else "en"
    rules = [r for r in load_rules(cfg.get("include_hidden", False)) if doc_type in r["applies_to"]]

    forced = [r for r in rules if r["id"] == cfg.get("force_rule")]
    pool = [r for r in rules if r not in forced]
    n_viol = rng.choice([0, 1, 1, 2, 2, 3])
    if forced:
        n_viol = max(0, n_viol - 2)
    violations = forced + rng.sample(pool, min(n_viol, len(pool)))
    pool = [r for r in pool if r not in violations]
    traps = rng.sample(pool, min(rng.choice([1, 1, 2]), len(pool)))
    pool = [r for r in pool if r not in traps]
    ambiguous = [rng.choice(pool)] if pool and rng.random() < 0.3 else []

    slots = ([{"kind": "violation", "rule_id": r["id"], "subtle": rng.random() < 0.5} for r in violations]
             + [{"kind": "trap", "rule_id": r["id"]} for r in traps]
             + [{"kind": "ambiguous", "rule_id": r["id"]} for r in ambiguous])
    for i, s in enumerate(slots, 1):
        s["slot"] = f"p{i}"
    return {"doc_id": doc_id, "split": split, "doc_type": doc_type, "language": language,
            "slots": slots, "n_neutral": rng.randint(5, 8),
            "numbering": rng.randrange(3), "noisy": cfg.get("noisy", False)}


# ---------- prompts ----------

def _generation_prompt(plan: dict, rules: dict, attempt: int) -> str:
    lang = "Arabic (Modern Standard, formal legal style)" if plan["language"] == "ar" else "English (formal legal style)"
    lines = []
    for s in plan["slots"]:
        r = rules[s["rule_id"]]
        if s["kind"] == "violation":
            tone = ("Make it SUBTLE: the softened, business-like wording a real company lawyer would use. "
                    "A careful reader can still tell it breaks the requirement, but it never announces it."
                    if s.get("subtle") else
                    "Make it a clear violation, but in realistic business language, not cartoonishly extreme.")
            how = (f"It must BREAK this requirement, in this way: {r['violation_looks_like']} {tone} "
                   f"Style reference only, write a different situation: {r['violation_example']}")
        elif s["kind"] == "trap":
            how = ("It must FULLY COMPLY with this requirement, but touch the same topic so it could look "
                   f"suspicious at first glance. Do not state that it complies with any law; it must read naturally. "
                   f"Style reference only, write a different situation: {r['trap_example']}")
        else:
            how = ("Write it so a careful lawyer could NOT tell whether it complies: vague wording, or a key "
                   "detail left undefined. It must be genuinely borderline, not a clear violation.")
        lines.append(f'- slot "{s["slot"]}" — requirement: {r["requirement"]}\n  {how}')
    planted = "\n".join(lines) if lines else "(none for this document)"
    return f"""You write realistic synthetic documents for testing a document review tool.
Write {DOC_TYPE_DESC[plan['doc_type']]}. Language: {lang}. Invent a fictional company name.

Planted clauses (each 1 to 3 sentences, self-contained, about its own topic only):
{planted}

Also write {plan['n_neutral']} neutral clauses about topics with NO link to personal data: fees, payment,
term and termination, liability, governing law, definitions, intellectual property, delivery, notices.
Neutral clauses must not mention personal data, privacy, consent, retention, deletion, transfers,
security incidents, health information or marketing.

Never mention any law name, regulation, article number, or the words violation, trap, compliance or test.
Do not number the clauses.

Return JSON only:
{{"title": "...", "planted": [{{"slot": "p1", "text": "..."}}], "neutral": ["...", "..."]}}
(variation {plan['doc_id']}-{attempt})"""


def _check_prompt(items: list[tuple[dict, str]], rules: dict) -> str:
    blocks = "\n\n".join(
        f'slot "{s["slot"]}"\nRequirement: {rules[s["rule_id"]]["requirement"]}\nClause: {text}'
        for s, text in items)
    return f"""For each clause, judge it ONLY against the requirement given with it.
verdict = "violation" if it clearly breaks the requirement, "compliant" if it clearly respects it,
"unclear" if a careful lawyer could not decide from the text.

{blocks}

Return JSON only: {{"results": [{{"slot": "p1", "verdict": "violation"}}]}}"""


def as_dict(data, list_key: str = "planted") -> dict:
    """Models sometimes return a bare list or a list wrapping the object; normalise to a dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict) and ("title" in data[0] or list_key in data[0]):
            return data[0]
        return {list_key: data}
    return {}


# ---------- assembly ----------

def _number(i: int, style: int, language: str) -> str:
    if language == "ar":
        n = str(i).translate(AR_DIGITS)
        return [f"{n}- ", f"المادة ({n}): ", f"{i}. "][style]
    return [f"{i}. ", f"Article {i}: ", f"({i}) "][style]


def add_noise(text: str, rng: random.Random, language: str) -> str:
    """Make clean text look like a messy copy-paste from a PDF."""
    out, since_page, page = [], 0, 2
    for ch in text:
        if language == "ar" and ch == "ة" and rng.random() < 0.3:
            ch = "ه"
        elif language == "ar" and ch == "ي" and rng.random() < 0.15:
            ch = "ى"
        if ch == " ":
            roll = rng.random()
            if roll < 0.05:
                ch = "\n"
            elif roll < 0.15:
                ch = "  "
            elif since_page > 500 and roll < 0.25:
                ch, since_page, page = f"\n- {page} -\n", 0, page + 1
        out.append(ch)
        since_page += 1
    return "".join(out)


def generate_doc(plan: dict, seed: int) -> dict:
    rules = {r["id"]: r for r in load_rules(include_hidden=True)}
    slots = plan["slots"]
    log = {"doc_id": plan["doc_id"], "attempts": 0, "dropped": []}
    kept: dict[str, str] = {}
    title, neutral = "", []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        log["attempts"] = attempt
        gen = as_dict(call(_generation_prompt(plan, rules, attempt), role="generator", json_mode=True).data)
        title = gen.get("title") or title
        neutral = [t for t in gen.get("neutral", []) if isinstance(t, str) and t.strip()] or neutral
        texts = {p.get("slot"): p.get("text", "").strip() for p in gen.get("planted", []) if isinstance(p, dict)}
        items = [(s, texts[s["slot"]]) for s in slots if texts.get(s["slot"])]
        verdicts = {}
        if items:
            res = as_dict(call(_check_prompt(items, rules), role="judge_eval", json_mode=True).data, "results")
            verdicts = {r.get("slot"): str(r.get("verdict", "")).lower() for r in res.get("results", [])}
        kept = {s["slot"]: t for s, t in items if verdicts.get(s["slot"]) == EXPECTED_VERDICT[s["kind"]]}
        if len(kept) == len(slots):
            break

    for s in slots:
        if s["slot"] not in kept:
            log["dropped"].append({"slot": s["slot"], "kind": s["kind"], "rule_id": s["rule_id"]})

    rng = random.Random(f"{seed}-{plan['doc_id']}-assemble")
    pieces = [{"text": kept[s["slot"]], "label": s["kind"], "rule_id": s["rule_id"],
               **({"subtle": s["subtle"]} if s["kind"] == "violation" else {})}
              for s in slots if s["slot"] in kept]
    pieces += [{"text": t, "label": "neutral", "rule_id": None} for t in neutral]
    rng.shuffle(pieces)

    clauses, body = [], []
    for i, p in enumerate(pieces, 1):
        clauses.append({"clause_id": f"c{i}", **p})
        body.append(_number(i, plan["numbering"], plan["language"]) + p["text"])
    full_text = f"{title}\n\n" + "\n\n".join(body)
    if plan["noisy"]:
        full_text = add_noise(full_text, rng, plan["language"])

    doc = {"doc_id": plan["doc_id"], "split": plan["split"], "doc_type": plan["doc_type"],
           "language": plan["language"], "title": title, "noisy": plan["noisy"],
           "clauses": clauses, "full_text": full_text}
    return {"doc": doc, "log": log}


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=CONFIG.get("seed", 42))
    args = ap.parse_args()

    out_dir = DATA_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = []
    for i in range(1, args.n + 1):
        doc_id = f"{args.split}_{i:03d}"
        if (out_dir / f"{doc_id}.json").exists():
            continue  # resume: never regenerate a finished document
        plans.append(plan_doc(random.Random(f"{args.seed}-{doc_id}"), args.split, doc_id))
    print(f"{args.split}: {args.n - len(plans)} already done, generating {len(plans)}")

    logs = []
    with ThreadPoolExecutor(max_workers=CONFIG.get("max_workers", 2)) as pool:
        futures = {pool.submit(generate_doc, p, args.seed): p["doc_id"] for p in plans}
        for fut in as_completed(futures):
            doc_id = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # keep going; a rerun picks the document up again
                print(f"  {doc_id} FAILED: {type(e).__name__}: {str(e)[:120]}")
                continue
            (out_dir / f"{doc_id}.json").write_text(
                json.dumps(result["doc"], ensure_ascii=False, indent=2), encoding="utf-8")
            logs.append(result["log"])
            labels = [c["label"] for c in result["doc"]["clauses"] if c["label"] != "neutral"]
            print(f"  {doc_id} ok  planted={labels}  dropped={len(result['log']['dropped'])}")

    with (out_dir / "_generation_log.jsonl").open("a", encoding="utf-8") as f:
        for entry in logs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    dropped = sum(len(e["dropped"]) for e in logs)
    print(f"done: {len(logs)} docs written, {dropped} planted clauses dropped by the quality check")
    print("llm totals:", TOTALS.summary())


if __name__ == "__main__":
    main()