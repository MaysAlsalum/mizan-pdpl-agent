"""Demo UI. Run from the project root:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rapidfuzz import fuzz  # noqa: E402

from src.agent import run as run_agent, segment  # noqa: E402
from src.catalog import rules_by_id  # noqa: E402

st.set_page_config(page_title="Mizan · PDPL Compliance Agent", page_icon=str(ROOT / "app" / "assets" / "mizan-mark.png"), layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
:root {--paper:#F6F3EC; --surface:#FFFFFF; --ink:#0F2233; --muted:#56636E; --line:#DDD6C8;
       --brand:#0B6E5C; --brand-tint:#DCEEE8; --violation:#B3361F; --violation-tint:#F7E0DA;
       --review:#8A5A00; --review-tint:#F8EBCB; --night:#0F2233; --gold:#8A5F10}
html, body, [class*="css"], .stMarkdown, button, input, textarea {font-family:"IBM Plex Sans Arabic","IBM Plex Sans",system-ui,sans-serif !important}
.hero {display:flex; gap:20px; align-items:center; background:var(--night); color:#EAF0F2; border-radius:16px; padding:24px 28px; margin-bottom:24px}
.hero h1 {margin:0; font-size:40px; line-height:1; font-weight:700; color:#EAF0F2}
.hero .ar {color:#45C4A7; font-weight:700; margin-inline-start:12px}
.hero p {margin:6px 0 0; color:#9AABB8; font-size:14px}
.clause {background:var(--surface); padding:12px 16px; margin:8px 0; border-radius:8px; border:1px solid var(--line); line-height:1.75; font-size:15px; color:var(--ink)}
.clause.violation {background:var(--violation-tint); border-color:var(--violation)}
.clause.escalate {background:var(--review-tint); border-color:var(--review)}
.clause.fixed {background:var(--brand-tint); border-color:var(--brand)}
.tag {font-size:12px; font-weight:600; padding:2px 8px; border-radius:4px; margin-inline-end:8px; background:var(--surface)}
.tag.v {color:var(--violation); border:1px solid var(--violation)} .tag.e {color:var(--review); border:1px solid var(--review)}
.rtl {direction:rtl; text-align:right}
[data-testid="stMetricValue"] {font-family:"IBM Plex Mono",monospace !important; color:var(--ink)}
[data-testid="stMetric"] {background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:12px 16px}
</style>""", unsafe_allow_html=True)

import base64  # noqa: E402

_mark = ROOT / "app" / "assets" / "mizan-mark.png"
_img = (f'<img src="data:image/png;base64,{base64.b64encode(_mark.read_bytes()).decode()}" width="64" '
        'style="background:#F6F3EC;border-radius:12px;padding:6px">') if _mark.exists() else ""
st.markdown(f"""<div class="hero">{_img}<div><h1>Mizan<span class="ar">ميزان</span></h1>
<p>Every clause, weighed. Finds Saudi PDPL violations clause by clause, cites the article, drafts a verified fix,
and escalates what it is unsure about. Assists a specialist; not legal advice.</p></div></div>""",
            unsafe_allow_html=True)

DEMO_SPLITS = ["robust_newtype", "robust_noisy", "robust_newrule", "dev"]  # never the test split


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("؀" <= c <= "ۿ" for c in letters) / len(letters) > 0.3


@st.cache_data
def sample_docs() -> dict[str, dict]:
    docs = {}
    for split in DEMO_SPLITS:
        for p in sorted((ROOT / "data" / split).glob("*.json")):
            if not p.name.startswith("_"):
                d = json.loads(p.read_text(encoding="utf-8"))
                docs[f"{d['doc_id']} · {d['doc_type']} · {d['language']}"] = d
    return docs


def clause_status(text: str, out: dict) -> tuple[str, list[str]]:
    tags, status = [], ""  # labels always carry the word, never colour alone
    for f in out["findings"]:
        if fuzz.partial_ratio(f["evidence"], text) >= 85:
            tags.append(f'<span class="tag v">{f["rule_id"]} · Violation</span>')
            status = "violation"
    for e in out["escalations"]:
        if fuzz.partial_ratio(e.get("evidence", ""), text) >= 85:
            tags.append(f'<span class="tag e">{e["rule_id"]} · Needs review</span>')
            status = status or "escalate"
    return status, tags


tab_review, tab_results = st.tabs(["Review a document", "Results"])

# ---------------- review ----------------
with tab_review:
    left, right = st.columns([1, 1])
    with left:
        source = st.radio("Document", ["Sample", "Paste text", "Upload .txt"], horizontal=True)
        doc = None
        if source == "Sample":
            docs = sample_docs()
            if docs:
                doc = docs[st.selectbox("Sample document", list(docs))]
            else:
                st.info("No sample documents found in data/.")
        elif source == "Paste text":
            text = st.text_area("Paste a contract, policy or form", height=260)
            if text.strip():
                doc = {"doc_id": "pasted", "full_text": text, "language": "ar" if is_arabic(text) else "en"}
        else:
            up = st.file_uploader("Text file", type=["txt"])
            if up:
                text = up.read().decode("utf-8", errors="ignore")
                doc = {"doc_id": up.name, "full_text": text, "language": "ar" if is_arabic(text) else "en"}
    with right:
        include_hidden = st.toggle("Include rule R08 (transfer outside the Kingdom)", value=False,
                                   help="Adaptation demo: R08 was hidden during development. "
                                        "Switching it on only adds a JSON entry, no code changes.")
        st.write("")
        go = st.button("Review document", type="primary", disabled=doc is None, use_container_width=True)

    if doc is not None and go:
        with st.status("Reviewing…", expanded=True) as status:
            st.write("Splitting into clauses and selecting the rules for each one")
            out = run_agent(doc, include_hidden=include_hidden)
            for line in out.get("trace", []):
                st.write(line)
            status.update(label=f"Done: {len(out['findings'])} violations, "
                                f"{len(out['escalations'])} sent for review", state="complete")
        st.session_state["last"] = (doc, out)

    if "last" in st.session_state:
        doc, out = st.session_state["last"]
        rtl = " rtl" if doc.get("language") == "ar" else ""
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Violations", len(out["findings"]))
        m2.metric("Needs specialist", len(out["escalations"]))
        m3.metric("Model calls", out["llm_calls"])
        m4.metric("Cost", f"${out['cost_usd']:.4f}")

        st.subheader("Document")
        for c in segment(doc["full_text"]):
            cls, tags = clause_status(c["text"], out)
            st.markdown(f'<div class="clause {cls}{rtl}">{"".join(tags)}{html.escape(c["text"])}</div>',
                        unsafe_allow_html=True)

        rules = rules_by_id(include_hidden=True)
        if out["findings"]:
            st.subheader("Violations and suggested fixes")
        for f in out["findings"]:
            r = rules.get(f["rule_id"], {})
            badge = {True: "✅ fix verified", False: "⚠️ fix not verified", None: ""}[f.get("fix_verified")]
            with st.expander(f"{f['rule_id']} · {r.get('title_en', '')} · {f.get('source', '')} · "
                             f"confidence {f['confidence']}  {badge}", expanded=True):
                st.markdown(f"**Why:** {f.get('reason', '')}")
                a, b = st.columns(2)
                a.markdown("**Original**")
                a.markdown(f'<div class="clause violation{rtl}">{html.escape(f.get("clause", f["evidence"]))}</div>',
                           unsafe_allow_html=True)
                if f.get("fix"):
                    b.markdown("**Suggested fix**")
                    b.markdown(f'<div class="clause fixed{rtl}">{html.escape(f["fix"])}</div>', unsafe_allow_html=True)

        if out["escalations"]:
            st.subheader("Sent to a specialist")
            st.dataframe(pd.DataFrame([{"Rule": e["rule_id"], "Reason": e["reason"],
                                        "Evidence": e.get("evidence", "")[:160]} for e in out["escalations"]]),
                         hide_index=True, use_container_width=True)

        st.download_button("Download report (JSON)", json.dumps(out, ensure_ascii=False, indent=2),
                           file_name=f"{out['doc_id']}_report.json", mime="application/json")

# ---------------- results ----------------
with tab_results:
    files = sorted((ROOT / "eval" / "results").glob("*.json"))
    runs = []
    for p in files:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if r.get("limit"):
            continue  # skip quick --limit runs
        name = r["system"] + (" +R08" if r.get("include_hidden") else "")
        runs.append({"System": name, "Split": r["split"], "file": p.name, **r["metrics"]})
    if not runs:
        st.info("No full evaluation runs yet. Run: python -m eval.evaluate --system agent --split dev")
    else:
        df = pd.DataFrame(runs).sort_values("file").groupby(["System", "Split"], as_index=False).last()
        split = st.selectbox("Split", sorted(df["Split"].unique()))
        view = df[df["Split"] == split]
        cols = ["System", "f1", "precision", "recall", "subtle_recall", "trap_fp_rate", "grounding_rate",
                "ambiguous_escalation_rate", "escalations_per_doc", "cost_per_doc_usd", "llm_calls_per_doc", "docs"]
        st.dataframe(view[[c for c in cols if c in view]], hide_index=True, use_container_width=True)

        per_rule = pd.DataFrame({row["System"]: row["per_rule_f1"] for _, row in view.iterrows()})
        if not per_rule.empty:
            st.markdown("**F1 per rule**")
            st.bar_chart(per_rule)

        log = ROOT / "eval" / "results" / "results_log.md"
        if log.exists():
            st.markdown("**Version history**")
            st.markdown(log.read_text(encoding="utf-8"))