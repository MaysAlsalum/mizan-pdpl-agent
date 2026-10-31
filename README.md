<p align="center"><img src="app/assets/mizan-mark.png" width="96" alt="Mizan"></p>

# Mizan | ميزان: PDPL Compliance Agent

**🎬 Demo video (3 min):** [Watch on Google Drive](https://drive.google.com/file/d/1f4iCfIge6U_NrvPn4X-HnOyDfh5Xkf7V/view?usp=sharing)

**📊 Slides:** [Mizan presentation (PDF)](docs/Mizan_presentation.pdf)

**An autonomous agent that reviews contracts, privacy policies and forms against Saudi Arabia's Personal Data Protection Law (PDPL), clause by clause.** It cites the article behind every verdict, backs it with an exact quote, drafts a fix and re-checks it, and sends low-confidence verdicts to a human specialist.

Built by **Mays Alsalum** for the HUMAIN × Misk *Future AI Builders* hackathon, **Challenge 3: Autonomous Enterprise**.

> Mizan assists a compliance specialist. It is not legal advice.

---

## Results on held-out test data

20 synthetic documents (Arabic and English) that were never opened or used for tuning. The system was frozen (tag `v3`) before this run.

| | Single-prompt baseline | **Mizan** |
|---|---|---|
| F1 | 0.864 | **0.952** |
| Precision | 0.792 | **0.909** |
| Recall | 0.950 | **1.000** |
| Recall on *subtle* violations | 0.917 | **1.000** |
| Compliant "trap" clauses wrongly flagged | 19.2% | **7.7%** |
| Evidence quotes found in the document | 100% | 100% |
| Cost per document | **$0.0016** | $0.016 |
| Model calls per document | **1.0** | 6.4 |

**Reading it:** the baseline flags almost one in five compliant-but-suspicious clauses. Mizan cuts that by 60% and misses no violation, including subtle ones written in soft business language. The price is about 10× the cost (still under 2 US cents per document). The baseline also dropped from F1 0.943 on the development set to 0.864 on test; Mizan stayed above 0.95.

### Robustness and adaptation

| Condition | Baseline F1 | Mizan F1 |
|---|---|---|
| Held-out test | 0.864 | **0.952** |
| Unseen document type (mobile app terms) | 0.800 (2 of 5 documents crashed on broken JSON) | **1.000** |
| Noisy text (PDF copy-paste: broken lines, page numbers, letter variants) | 1.000 | 1.000 |
| **New rule added at test time, no code change** (R08, cross-border transfer) | — | 0.286 → **1.000** |

**Noise** did not hurt either system on these 5 documents, so this split does not separate them; a larger and noisier set would be needed. **New document type:** the baseline crashed on 2 of 5 documents and missed every R03 and R05 violation it did read; Mizan found all of them. Robustness splits have 5 documents each, so treat them as indicative.

**Adaptation:** rule R08 was hidden from the catalog during all development. With it absent, Mizan found 17% of the violations in those documents (R08 recall 0). Adding R08 is a single JSON entry in `rules/catalog.json`; with it, F1 rose to 1.0 and R08 recall to 1.0, with no change to the code.

---

## How it works

<p align="center"><b>Document → split into clauses → gate → judge with evidence → fix and re-verify → report or escalate</b></p>

1. **Split (no model).** Clauses are split on numbering that runs in sequence (`1.`, `١-`, `المادة (٢)`, `Article 3`), so stray numbers like "(60) days" and page markers in noisy text do not break the split.
2. **Gate (no model).** A clause that never touches personal data (fees, courts, IP) is skipped, so the model only reads what matters. The gate knows personal-data *fields* (national ID, marital status, family members, workplace…) in Arabic and English, not just the word "data".
3. **Judge with evidence.** Each remaining clause is checked against the rule catalog with the document title as context, six clauses per call. Each rule in the prompt carries what a violation looks like *and* a compliant example that looks suspicious, which cuts false alarms. Every verdict returns a confidence and an exact quote.
4. **Grounding check (no model).** The quote must exist in the clause (fuzzy match ≥ 90), otherwise the finding is re-anchored or escalated. Mizan cannot cite text that is not there.
5. **Fix and re-verify.** Each confirmed violation is rewritten, and the rewrite goes back through the judge. A fix that still fails after two attempts is escalated with its draft.
6. **Escalate instead of guessing.** Low confidence or "unclear" verdicts go to a human specialist with the reason, instead of being reported as violations or retried forever.

The **rule catalog** (`rules/catalog.json`) holds 11 PDPL obligations, each with its article number from the official text, the requirement, what a violation looks like, and three examples (compliant, violation, trap). The same file drives the agent and the synthetic data generator.

---

## How we got here: measured, diagnosed, fixed

| Version | Change | Dev F1 | Dev recall | Why |
|---|---|---|---|---|
| v0 | Single prompt, whole document (baseline) | 0.943 | 0.943 | Strong on clear cases; 2 of 30 documents crashed on broken JSON |
| v1 | Agent, top-3 rules per clause by keyword | 0.727* | 0.571* | Subtle violations avoid the obvious keywords, so the right rule was never sent |
| v2 | Gate + every rule for the guessed document type | 0.913 | 0.887 | Error analysis (`eval/analyze.py`): all 6 misses came from rule selection, none from the judge |
| v3 | Gate knows personal-data fields; all rules; document context; second review off | **0.990** | **0.981** | Frozen. Then run once on test |

\* v1 was measured on 3 documents.

**Ablation that changed the design:** a second review of borderline verdicts sounded safer, but with the same model as the judge it *added* false alarms (dev F1 0.913 with it, 0.931 without). It is now off by default (`agent:with_recheck` turns it back on).

---

## Data and evaluation

**Synthetic data with planted labels.** Python decides each document's plan (type, language, which rules to violate, which traps and ambiguous clauses to plant); the model only writes the text, so the correct answer is known by construction. Half the violations are written to be *subtle*. A second model call checks every planted clause, and clauses that do not read as intended are regenerated or dropped (logged in `data/<split>/_generation_log.jsonl`).

| Split | Documents | Purpose |
|---|---|---|
| `dev` | 30 | Development and error analysis |
| `test` | 20 | Held out, run once per frozen version |
| `robust_newtype` | 5 | Mobile app terms, a type never seen in dev/test |
| `robust_noisy` | 5 | PDF-style noise |
| `robust_newrule` | 5 | Contains the hidden rule R08 |

**Scoring (`eval/metrics.py`).** A finding counts only if the rule is right **and** the evidence points at the planted clause, so guessing the right rule from the wrong clause is a false positive. Flagging a genuinely ambiguous clause is neither rewarded nor punished. A second, unlabelled rule on an already-violating clause is counted as unverified rather than wrong.

---

## Run it

Requires Python 3.10+ and a Gemini or OpenAI API key.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                 # add GEMINI_API_KEY or OPENAI_API_KEY
```

**Reproduce every number above at zero cost:** every model call is cached in `.cache/` (committed), so re-running reads the cache instead of calling the API.

```bash
python -m pytest -q tests                                   # 14 tests
python -m eval.evaluate --system baseline_small --split test
python -m eval.evaluate --system agent --split test
python -m eval.evaluate --system agent --split robust_newrule --include-hidden
python -m eval.analyze                                      # error analysis on the latest dev run
```

**Demo app:**

```bash
streamlit run app/streamlit_app.py
```

Models are set in `config.yaml` (default: `gemini-3.5-flash-lite`; any LiteLLM model name works).

## Project layout

```
rules/catalog.json      PDPL rule catalog (11 rules, article numbers)
src/agent.py            the agent (split, gate, judge, ground, fix, escalate)
src/baseline.py         single-prompt baseline
src/generate_data.py    synthetic data with planted labels
src/llm.py              every model call: cache, rate limit, retries, cost
eval/evaluate.py        run a system on a split, save every prediction
eval/metrics.py         scoring
eval/analyze.py         error analysis: why each violation was missed
app/streamlit_app.py    demo UI
tests/                  unit tests for scoring and the agent's non-model steps
```

## Limitations

- **Escalation is not yet proven.** Low-confidence and "unclear" verdicts are escalated, but the judge was confident about the planted *ambiguous* clauses, so none of them were escalated on dev or test (`ambiguous_escalation_rate` = 0). Escalating borderline clauses, not only low-confidence ones, is the first next step.
- **One model family throughout.** After the free-tier switch, `gemini-3.5-flash-lite` generated the data, ran the quality check, and is the model under test (for both Mizan and the baseline). This can flatter both systems; an independent model family for generation and checking would make the labels stronger.
- **Small samples, one run.** 20 test documents and 5 per robustness split, each run once, with no confidence intervals. Treat the differences as indicative.
- **Synthetic data.** Documents are generated, not real contracts. The quality check and the robustness splits reduce, but do not remove, that gap. Next step: validate on real documents with compliance specialists.
- **Some traps are debatable** (for example, asking for bank details on a job application before hiring).
- **11 obligations, not the whole law.** Rules that need knowledge outside the document (registration, DPO appointment) are out of scope.
- **Cost and latency.** Mizan costs about 10× the baseline per document; on the free API tier, latency is dominated by rate limiting.
- **Rule references** were checked against the official English translation of the PDPL; the 72-hour breach deadline comes from the Implementing Regulations.

## Sources

- [Personal Data Protection Law, official Arabic text (Bureau of Experts)](https://laws.boe.gov.sa/BoeLaws/Laws/LawDetails/b7cfae89-828e-4994-b167-adaa00e37188/1)
- [Personal Data Protection Law, English translation](https://misa.gov.sa/app/uploads/2025/08/PersonalDataProtectionLaw.pdf)

