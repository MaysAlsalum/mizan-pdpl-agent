"""Load the PDPL rule catalog. The hidden rule stays out until the adaptation test."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "rules" / "catalog.json"


def load_rules(include_hidden: bool = False, path: Path = CATALOG_PATH) -> list[dict]:
    """Return the rules; rules marked hidden_during_dev are skipped unless include_hidden."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in data["rules"] if include_hidden or not r.get("hidden_during_dev")]


def rules_by_id(include_hidden: bool = False) -> dict[str, dict]:
    return {r["id"]: r for r in load_rules(include_hidden)}


def rules_for_doc_type(doc_type: str, include_hidden: bool = False) -> list[dict]:
    """Rules whose applies_to covers this document type (unknown types get every rule)."""
    rules = load_rules(include_hidden)
    matching = [r for r in rules if doc_type in r["applies_to"]]
    return matching or rules


if __name__ == "__main__":
    visible, everything = load_rules(), load_rules(include_hidden=True)
    print(f"{len(visible)} visible rules, {len(everything) - len(visible)} hidden")
    for r in everything:
        flag = " (hidden)" if r.get("hidden_during_dev") else ""
        print(f"  {r['id']}  {r['title_en']:32} {r['source']}{flag}")