"""Demo UI. Run from the project root:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import base64
import re
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
from src.llm import call, model_for  # noqa: E402

st.set_page_config(page_title="Mizan · PDPL Compliance Agent",
                   page_icon=str(ROOT / "app" / "assets" / "mizan-mark.png"), layout="wide")

# ---------------- language ----------------
T = {
    "en": {
        "tagline": "Every clause, weighed. Finds Saudi PDPL violations clause by clause, cites the article, "
                   "drafts a verified fix, and escalates what it is unsure about. Assists a specialist; not legal advice.",
        "tab_review": "Review a document", "tab_results": "Results",
        "document": "Document", "sample": "Sample", "paste": "Paste text", "upload": "Upload .txt",
        "sample_doc": "Sample document", "no_samples": "No sample documents found in data/.",
        "paste_label": "Paste a contract, policy or form", "file_label": "Text file",
        "r08": "Include rule R08 (transfer outside the Kingdom)",
        "r08_help": "Adaptation demo: R08 was hidden during development. Switching it on only adds a JSON entry, no code changes.",
        "review": "Review document", "reviewing": "Reviewing…",
        "step1": "Splitting into clauses and selecting the rules for each one",
        "done": "Done: {v} violations, {e} sent for review",
        "m_viol": "Violations", "m_esc": "Needs specialist", "m_calls": "Model calls", "m_cost": "Cost",
        "tag_v": "Violation", "tag_e": "Needs review",
        "fixes": "Violations and suggested fixes", "confidence": "confidence",
        "verified": "✅ fix verified", "not_verified": "⚠️ fix not verified",
        "why": "Why", "original": "Original", "fix": "Suggested fix",
        "escalated": "Sent to a specialist", "c_rule": "Rule", "c_reason": "Reason", "c_evidence": "Evidence",
        "download": "Download report (JSON)",
        "no_runs": "No full evaluation runs yet. Run: python -m eval.evaluate --system agent --split dev",
        "split": "Split", "per_rule": "F1 per rule", "history": "Version history",
        "lang_btn": "عربي", "dark": "Dark mode",
        "err_quota": "The model API's daily quota is used up, so this document could not be reviewed now.",
        "err_other": "The review could not finish because the model API returned an error.",
        "err_hint": "Documents already reviewed open instantly from the cache. Try one of those, or try again later.",
    },
    "ar": {
        "tagline": "كل بند، في ميزانه. يكتشف مخالفات نظام حماية البيانات الشخصية بنداً بنداً، ويستشهد بالمادة، "
                   "ويقترح تعديلاً موثّقاً، ويحيل ما لا يتأكد منه إلى مختص. أداة مساعدة للمختص، وليست استشارة قانونية.",
        "tab_review": "مراجعة مستند", "tab_results": "النتائج",
        "document": "المستند", "sample": "مستند تجريبي", "paste": "لصق نص", "upload": "رفع ملف txt",
        "sample_doc": "اختر مستنداً", "no_samples": "لا توجد مستندات تجريبية في مجلد data.",
        "paste_label": "الصق عقداً أو سياسة أو نموذجاً", "file_label": "ملف نصي",
        "r08": "تضمين القاعدة R08 (نقل البيانات خارج المملكة)",
        "r08_help": "عرض التكيّف: القاعدة R08 كانت مخفية أثناء التطوير. تفعيلها يضيف سطراً في ملف JSON فقط، دون أي تعديل في الكود.",
        "review": "افحص المستند", "reviewing": "جارٍ الفحص…",
        "step1": "تقسيم المستند إلى بنود واختيار القواعد لكل بند",
        "done": "تم: {v} مخالفات، و{e} محالة للمراجعة",
        "m_viol": "المخالفات", "m_esc": "تحتاج مختصاً", "m_calls": "استدعاءات النموذج", "m_cost": "التكلفة",
        "tag_v": "مخالفة", "tag_e": "تحتاج مراجعة",
        "fixes": "المخالفات والتعديلات المقترحة", "confidence": "الثقة",
        "verified": "✅ تعديل موثّق", "not_verified": "⚠️ تعديل غير موثّق",
        "why": "السبب", "original": "النص الأصلي", "fix": "التعديل المقترح",
        "escalated": "محالة إلى مختص", "c_rule": "القاعدة", "c_reason": "السبب", "c_evidence": "الدليل",
        "download": "تحميل التقرير (JSON)",
        "no_runs": "لا توجد نتائج تقييم كاملة بعد. شغّل: python -m eval.evaluate --system agent --split dev",
        "split": "مجموعة البيانات", "per_rule": "F1 لكل قاعدة", "history": "سجل النسخ",
        "lang_btn": "English", "dark": "الوضع الليلي",
        "err_quota": "انتهت الحصة اليومية لواجهة النموذج، فتعذّر فحص هذا المستند الآن.",
        "err_other": "تعذّر إكمال الفحص بسبب خطأ من واجهة النموذج.",
        "err_hint": "المستندات التي فُحصت سابقاً تُفتح فوراً من الذاكرة المؤقتة. جرّب أحدها، أو أعد المحاولة لاحقاً.",
    },
}
st.session_state.setdefault("lang", "en")
st.session_state.setdefault("dark", False)

bar_l, bar_r1, bar_r2 = st.columns([7, 1.6, 1.2], vertical_alignment="center")
with bar_r1:
    st.session_state["dark"] = st.toggle(T[st.session_state["lang"]]["dark"], value=st.session_state["dark"])
with bar_r2:
    if st.button(T[st.session_state["lang"]]["lang_btn"], use_container_width=True):
        st.session_state["lang"] = "ar" if st.session_state["lang"] == "en" else "en"
        st.rerun()

LANG, DARK = st.session_state["lang"], st.session_state["dark"]
t = T[LANG]

# ---------------- theme (Mizan tokens, light and dark) ----------------
P = ({"paper": "#0D1822", "surface": "#14242F", "ink": "#EAF0F2", "muted": "#9AABB8", "line": "#2A3B4A",
      "brand": "#45C4A7", "brand_tint": "#123A33", "violation": "#F2836D", "violation_tint": "#3D1E18",
      "review": "#F0BE55", "review_tint": "#3A2E12", "on_brand": "#0D1822"} if DARK else
     {"paper": "#F6F3EC", "surface": "#FFFFFF", "ink": "#0F2233", "muted": "#56636E", "line": "#DDD6C8",
      "brand": "#0B6E5C", "brand_tint": "#DCEEE8", "violation": "#B3361F", "violation_tint": "#F7E0DA",
      "review": "#8A5A00", "review_tint": "#F8EBCB", "on_brand": "#FFFFFF"})

FONT = '"IBM Plex Sans Arabic","IBM Plex Sans",system-ui,sans-serif'

direction_css = """
[data-testid="stMainBlockContainer"], .main .block-container {direction:rtl}
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], .stApp label, .stApp p, .stApp h1, .stApp h2, .stApp h3 {text-align:right}
.hero, .hero p, .hero h1 {text-align:right}
""" if LANG == "ar" else ""

dark_css = f"""
.stApp, [data-testid="stHeader"], [data-testid="stAppViewContainer"], [data-testid="stBottomBlockContainer"] {{background:{P['paper']} !important}}
.stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp li, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{color:{P['ink']}}}
div[data-baseweb="select"] > div, div[data-baseweb="select"] div, div[data-baseweb="input"], div[data-baseweb="input"] input, div[data-baseweb="textarea"], div[data-baseweb="textarea"] textarea, [data-testid="stFileUploaderDropzone"] {{background:{P['surface']} !important; color:{P['ink']} !important; border-color:{P['line']} !important}}
div[data-baseweb="select"] svg {{fill:{P['muted']} !important}}
div[data-baseweb="popover"], div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="menu"] {{background:{P['surface']} !important; color:{P['ink']} !important}}
div[data-baseweb="popover"] li:hover {{background:{P['line']} !important}}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary, [data-testid="stStatusWidget"] {{background:{P['surface']} !important; border-color:{P['line']} !important}}
[data-testid="stRadio"] [role="radio"] > div:first-child {{border-color:{P['muted']} !important}}
""" if DARK else ""

CSS = f"""
html, body, .stApp, .stApp button, .stApp input, .stApp textarea, .stApp label, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3 {{font-family:{FONT} !important}}
.stApp {{background:{P['paper']}}}
[data-testid="stMainBlockContainer"] {{max-width:1240px; padding-top:5rem}}
.stApp p, .stApp li, .stApp label, [data-testid="stWidgetLabel"] p {{font-size:18px !important; line-height:1.6}}
[data-testid="stWidgetLabel"] p {{font-weight:600; color:{P['ink']} !important}}
.stApp h2 {{font-size:30px !important; font-weight:700 !important; color:{P['ink']} !important; margin-top:1.5rem}}
.stApp h3 {{font-size:24px !important; font-weight:700 !important; color:{P['ink']} !important}}
.hero {{display:flex; gap:28px; align-items:center; background:#0F2233; border-radius:20px; padding:36px 40px; margin:8px 0 28px; border:1px solid {P['line']}}}
.hero h1 {{margin:0; font-size:56px !important; line-height:1; font-weight:700; color:#EAF0F2 !important}}
.hero .ar {{color:#45C4A7 !important; font-weight:700}}
.hero .sep {{color:#56708A !important; font-weight:400; margin:0 14px}}
.hero p {{margin:12px 0 0; color:#9AABB8 !important; font-size:18px !important; max-width:860px}}
.stTabs [data-baseweb="tab-list"] {{gap:28px; border-bottom:1px solid {P['line']}}}
.stTabs [data-baseweb="tab"] p {{font-size:20px !important; font-weight:600; color:{P['muted']} !important}}
.stTabs [aria-selected="true"] p {{color:{P['brand']} !important}}
.stTabs [data-baseweb="tab-highlight"] {{background:{P['brand']} !important; height:3px}}
.stButton button, .stDownloadButton button {{border-radius:10px !important; min-height:52px; font-size:18px !important; font-weight:600 !important}}
.stButton button p, .stDownloadButton button p {{font-size:18px !important; font-weight:600 !important}}
.stButton button[kind="primary"] {{background:{P['brand']} !important; border:none !important}}
.stButton button[kind="primary"] p {{color:{P['on_brand']} !important}}
.stButton button[kind="primary"]:disabled {{opacity:.45}}
.stButton button[kind="secondary"], .stDownloadButton button {{background:{P['surface']} !important; border:1px solid {P['line']} !important}}
.stButton button[kind="secondary"] p, .stDownloadButton button p {{color:{P['ink']} !important}}
div[data-baseweb="select"] > div {{border-radius:10px !important; min-height:52px; font-size:17px}}
.clause {{background:{P['surface']}; padding:16px 20px; margin:10px 0; border-radius:10px; border:1px solid {P['line']}; line-height:1.85; font-size:18px; color:{P['ink']} !important}}
.clause.violation {{background:{P['violation_tint']}; border-color:{P['violation']}; border-width:2px}}
.clause.escalate {{background:{P['review_tint']}; border-color:{P['review']}; border-width:2px}}
.clause.fixed {{background:{P['brand_tint']}; border-color:{P['brand']}; border-width:2px}}
.tag {{display:inline-block; font-size:14px; font-weight:700; padding:2px 10px; border-radius:6px; margin-inline-end:10px; background:{P['surface']}}}
.tag.v {{color:{P['violation']} !important; border:1px solid {P['violation']}}}
.tag.e {{color:{P['review']} !important; border:1px solid {P['review']}}}
.rtl {{direction:rtl; text-align:right}} .ltr {{direction:ltr; text-align:left}}
[data-testid="stMetric"] {{background:{P['surface']}; border:1px solid {P['line']}; border-radius:16px; padding:20px 24px}}
[data-testid="stMetricLabel"] p {{color:{P['muted']} !important; font-size:16px !important; font-weight:600}}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{font-family:"IBM Plex Mono",monospace !important; color:{P['ink']} !important; font-size:40px !important}}
[data-testid="stExpander"] summary p {{font-size:18px !important; font-weight:600}}
{dark_css}
{direction_css}
"""
# A blank line inside st.markdown ends the HTML block and prints the rest as text, so drop blank lines.
CSS = "\n".join(line for line in CSS.splitlines() if line.strip())
st.markdown('<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700'
            '&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">\n<style>\n' + CSS + '\n</style>',
            unsafe_allow_html=True)

_mark = ROOT / "app" / "assets" / "mizan-mark.png"
_img = (f'<img src="data:image/png;base64,{base64.b64encode(_mark.read_bytes()).decode()}" width="88" '
        'style="background:#F6F3EC;border-radius:16px;padding:8px">') if _mark.exists() else ""
st.markdown(f'<div class="hero">{_img}<div><h1>Mizan<span class="sep">|</span><span class="ar">ميزان</span></h1>'
            f'<p>{t["tagline"]}</p></div></div>', unsafe_allow_html=True)

DEMO_SPLITS = ["robust_newtype", "robust_noisy", "robust_newrule", "dev"]  # never the test split


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("؀" <= c <= "ۿ" for c in letters) / len(letters) > 0.3


TRANSLATE_MODELS = [model_for("small"), "gemini/gemini-3.5-flash"]  # each Gemini model has its own daily quota


@st.cache_data(show_spinner=False)
def _translate(texts: tuple[str, ...]) -> tuple[str, ...]:
    """One call for all reasons of a document. Raises on failure so that a failure is NOT cached."""
    prompt = ("Translate each compliance reviewer's note into clear Modern Standard Arabic. Keep rule ids "
              "(R01), article numbers and the abbreviation PDPL as they are.\n"
              'Return JSON only: {"translations": ["...", "..."]}, same order and count as the input.\n\n'
              + json.dumps(list(texts), ensure_ascii=False))
    last = None
    for model in dict.fromkeys(TRANSLATE_MODELS):
        try:
            res = call(prompt, model=model, json_mode=True, max_retries=1)
            out = res.data.get("translations", []) if isinstance(res.data, dict) else res.data
            if isinstance(out, list) and len(out) == len(texts) and all(isinstance(x, str) for x in out):
                return tuple(out)
            last = ValueError("wrong number of translations")
        except Exception as e:  # quota, rate limit, bad JSON: try the next model
            last = e
    raise RuntimeError(f"translation unavailable: {last}")


def to_arabic(texts: tuple[str, ...]) -> tuple[tuple[str, ...], bool]:
    todo = tuple(dict.fromkeys(x for x in texts if x and not is_arabic(x)))
    if not todo:
        return texts, True
    try:
        lookup = dict(zip(todo, _translate(todo)))
        return tuple(lookup.get(x, x) for x in texts), True
    except Exception:
        return texts, False


ESC_AR = [
    (r"low confidence \(([\d.]+)\)", r"ثقة منخفضة (\1)"),
    (r"judged unclear", "حكم غير واضح"),
    (r"evidence not found in text", "الدليل غير موجود في النص"),
    (r"fix failed after (\d+) attempts", r"فشل التعديل بعد \1 محاولات"),
    (r"reviewers disagree or unsure", "اختلاف المراجعين أو عدم تأكدهم"),
]


def esc_reason(reason: str) -> str:
    if LANG != "ar":
        return reason
    for pat, rep in ESC_AR:
        reason = re.sub(pat, rep, reason)
    return reason


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
            tags.append(f'<span class="tag v">{f["rule_id"]} · {t["tag_v"]}</span>')
            status = "violation"
    for e in out["escalations"]:
        if fuzz.partial_ratio(e.get("evidence", ""), text) >= 85:
            tags.append(f'<span class="tag e">{e["rule_id"]} · {t["tag_e"]}</span>')
            status = status or "escalate"
    return status, tags


tab_review, tab_results = st.tabs([t["tab_review"], t["tab_results"]])

# ---------------- review ----------------
with tab_review:
    left, right = st.columns([1, 1])
    with left:
        choices = {"sample": t["sample"], "paste": t["paste"], "upload": t["upload"]}
        source = st.radio(t["document"], list(choices), format_func=choices.get, horizontal=True)
        doc = None
        if source == "sample":
            docs = sample_docs()
            if docs:
                doc = docs[st.selectbox(t["sample_doc"], list(docs))]
            else:
                st.info(t["no_samples"])
        elif source == "paste":
            text = st.text_area(t["paste_label"], height=260)
            if text.strip():
                doc = {"doc_id": "pasted", "full_text": text, "language": "ar" if is_arabic(text) else "en"}
        else:
            up = st.file_uploader(t["file_label"], type=["txt"])
            if up:
                text = up.read().decode("utf-8", errors="ignore")
                doc = {"doc_id": up.name, "full_text": text, "language": "ar" if is_arabic(text) else "en"}
    with right:
        include_hidden = st.toggle(t["r08"], value=False, help=t["r08_help"])
        st.write("")
        go = st.button(t["review"], type="primary", disabled=doc is None, use_container_width=True)

    if doc is not None and go:
        out = None
        with st.status(t["reviewing"], expanded=True) as status:
            st.write(t["step1"])
            try:
                out = run_agent(doc, include_hidden=include_hidden)
            except Exception as e:  # most often the API's daily quota; never show a raw traceback
                quota = "quota" in str(e).lower() or "429" in str(e)
                status.update(label=t["err_quota"] if quota else t["err_other"], state="error")
                st.write(t["err_hint"])
            if out:
                for line in out.get("trace", []):
                    st.write(line)
                status.update(label=t["done"].format(v=len(out["findings"]), e=len(out["escalations"])),
                              state="complete")
        if out:
            st.session_state["last"] = (doc, out)

    if "last" in st.session_state:
        doc, out = st.session_state["last"]
        dirc = " rtl" if doc.get("language") == "ar" else " ltr"   # clause text follows the document's language
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t["m_viol"], len(out["findings"]))
        m2.metric(t["m_esc"], len(out["escalations"]))
        m3.metric(t["m_calls"], out["llm_calls"])
        m4.metric(t["m_cost"], f"${out['cost_usd']:.4f}")

        st.markdown(f"## {t['document']}")
        for c in segment(doc["full_text"]):
            cls, tags = clause_status(c["text"], out)
            st.markdown(f'<div class="clause {cls}{dirc}">{"".join(tags)}{html.escape(c["text"])}</div>',
                        unsafe_allow_html=True)

        rules = rules_by_id(include_hidden=True)
        reasons = tuple(f.get("reason", "") for f in out["findings"])
        reasons_ar = {}
        if LANG == "ar" and reasons:
            with st.spinner("جارٍ ترجمة الأسباب…"):
                translated, ok = to_arabic(reasons)
                reasons_ar = dict(zip(reasons, translated))
            if not ok:
                st.caption("تعذّرت ترجمة الأسباب الآن (حدود الاستخدام في واجهة النموذج)، فتظهر بالإنجليزية.")
        if out["findings"]:
            st.markdown(f"## {t['fixes']}")
        for f in out["findings"]:
            r = rules.get(f["rule_id"], {})
            title = r.get("title_ar" if LANG == "ar" else "title_en", "")
            badge = {True: t["verified"], False: t["not_verified"], None: ""}[f.get("fix_verified")]
            with st.expander(f"{f['rule_id']} · {title} · {f.get('source', '')} · "
                             f"{t['confidence']} {f['confidence']}  {badge}", expanded=True):
                why = reasons_ar.get(f.get("reason", ""), f.get("reason", ""))
                st.markdown(f"**{t['why']}:** {why}")
                a, b = st.columns(2)
                a.markdown(f"**{t['original']}**")
                a.markdown(f'<div class="clause violation{dirc}">{html.escape(f.get("clause", f["evidence"]))}</div>',
                           unsafe_allow_html=True)
                if f.get("fix"):
                    b.markdown(f"**{t['fix']}**")
                    b.markdown(f'<div class="clause fixed{dirc}">{html.escape(f["fix"])}</div>', unsafe_allow_html=True)

        if out["escalations"]:
            st.markdown(f"## {t['escalated']}")
            st.dataframe(pd.DataFrame([{t["c_rule"]: e["rule_id"], t["c_reason"]: esc_reason(e["reason"]),
                                        t["c_evidence"]: e.get("evidence", "")[:160]} for e in out["escalations"]]),
                         hide_index=True, use_container_width=True)

        st.download_button(t["download"], json.dumps(out, ensure_ascii=False, indent=2),
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
        if r.get("limit") or r.get("errors", 0) >= r.get("metrics", {}).get("docs", 1):
            continue  # skip quick --limit runs and runs where every document failed
        name = r["system"] + (" +R08" if r.get("include_hidden") else "")
        m = dict(r["metrics"])
        rule_f1 = m.pop("per_rule_f1", {})
        runs.append({"System": name, "Split": r["split"], "file": p.name, "_rules": json.dumps(rule_f1), **m})
    if not runs:
        st.info(t["no_runs"])
    else:
        df = pd.DataFrame(runs).sort_values("file").groupby(["System", "Split"], as_index=False).last()
        split = st.selectbox(t["split"], sorted(df["Split"].unique()))
        view = df[df["Split"] == split]
        cols = ["System", "f1", "precision", "recall", "subtle_recall", "trap_fp_rate", "grounding_rate",
                "ambiguous_escalation_rate", "escalations_per_doc", "cost_per_doc_usd", "llm_calls_per_doc", "docs"]
        st.dataframe(view[[c for c in cols if c in view]], hide_index=True, use_container_width=True)

        per_rule = pd.DataFrame({row["System"]: json.loads(row["_rules"]) for _, row in view.iterrows()}).astype(float)
        if not per_rule.empty:
            st.markdown(f"**{t['per_rule']}**")
            st.bar_chart(per_rule)

        log = ROOT / "eval" / "results" / ("results_log.ar.md" if LANG == "ar" else "results_log.md")
        if log.exists():
            st.markdown(f"**{t['history']}**")
            st.markdown(log.read_text(encoding="utf-8"))