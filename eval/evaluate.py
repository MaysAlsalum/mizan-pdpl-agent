"""Run a system on a data split, score it, and save every prediction plus the metrics.

Usage:
    python -m eval.evaluate --system baseline_small --split dev
    python -m eval.evaluate --system baseline_small --split dev --limit 3
    python -m eval.evaluate --system agent --split robust_newrule --include-hidden
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from eval.metrics import aggregate, score_doc
from src.llm import CONFIG, TOTALS

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"


def get_system(name: str):
    """Map a system name to a function doc -> output. Imports lazily so a missing module only fails its own system."""
    if name in ("baseline_small", "baseline_large"):
        from src.baseline import run
        role = name.split("_")[1]
        return lambda doc, hidden: run(doc, role=role, include_hidden=hidden)
    if name.startswith("agent"):
        from src.agent import run  # added in phase 4
        variant = name.split(":", 1)[1] if ":" in name else None
        return lambda doc, hidden: run(doc, include_hidden=hidden, ablate=variant)
    raise SystemExit(f"unknown system: {name}")


def load_split(split: str, limit: int | None = None) -> list[dict]:
    files = sorted(p for p in (ROOT / "data" / split).glob("*.json") if not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"no documents in data/{split}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in files[:limit]]


def print_report(name: str, split: str, m: dict) -> None:
    print(f"\n{name} on {split}  ({m['docs']} docs)")
    for k in ("f1", "precision", "recall", "subtle_recall", "trap_fp_rate", "grounding_rate",
              "ambiguous_escalation_rate", "escalations_per_doc",
              "cost_per_doc_usd", "latency_per_doc_s", "llm_calls_per_doc"):
        print(f"  {k:28} {m[k]}")
    print("  per-rule F1:", "  ".join(f"{r}={v}" for r, v in m["per_rule_f1"].items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, help="only the first N documents (quick checks)")
    ap.add_argument("--include-hidden", action="store_true", help="add the hidden rule (adaptation test)")
    args = ap.parse_args()

    system = get_system(args.system)
    docs = load_split(args.split, args.limit)
    start = time.perf_counter()

    def one(doc: dict) -> dict:
        try:
            return system(doc, args.include_hidden)
        except Exception as e:  # a crash counts as an empty answer, and is recorded
            return {"doc_id": doc["doc_id"], "findings": [], "escalations": [],
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    with ThreadPoolExecutor(max_workers=CONFIG.get("max_workers", 2)) as pool:
        outputs = list(pool.map(one, docs))

    scores = [score_doc(d, o) for d, o in zip(docs, outputs)]
    metrics = aggregate(scores)
    errors = [o for o in outputs if "error" in o]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = args.system.replace(":", "-") + ("_hidden" if args.include_hidden else "")
    out_path = RESULTS_DIR / f"{tag}_{args.split}_{stamp}.json"
    out_path.write_text(json.dumps({
        "system": args.system, "split": args.split, "include_hidden": args.include_hidden,
        "limit": args.limit, "models": CONFIG["models"], "metrics": metrics,
        "errors": len(errors), "wall_time_s": round(time.perf_counter() - start, 1),
        "llm_totals": TOTALS.summary(),
        "per_doc": [{"doc_id": d["doc_id"], "score": s, "output": o} for d, s, o in zip(docs, scores, outputs)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(args.system, args.split, metrics)
    if errors:
        print(f"  !! {len(errors)} documents failed, first error: {errors[0]['error']}")
    print(f"  saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()