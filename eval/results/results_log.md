| Version | Change | Data | F1 | Recall | Subtle recall | Traps flagged | What we learned |
|---|---|---|---|---|---|---|---|
| v0 | Baseline: one prompt, whole document | dev, 3 docs | 0.933 | 1.000 | 1.000 | 0.0% | Reference point |
| v1 | Agent: top 3 rules per clause, by keyword | dev, 3 docs | 0.727 | 0.571 | 0.400 | 0.0% | Subtle wording avoids the keywords, so the right rule never reached the judge |
| v2 | Personal-data gate + every rule for the document type | dev, 3 docs | 1.000 | 1.000 | 1.000 | 0.0% | Fixed the 3-document check; needed the full set to confirm |
| v0 | Baseline | dev, 30 docs | 0.943 | 0.943 | 0.964 | 5.0% | 2 of 30 documents crashed on broken JSON |
| v2 | Same agent, full set | dev, 30 docs | 0.913 | 0.887 | 0.857 | 2.5% | Error analysis: all 6 misses came from rule selection, none from the judge |
| v2 − recheck | Ablation: second review switched off | dev, 30 docs | 0.931 | 0.887 | 0.857 | 2.5% | With the same model as the judge, the second review only added false alarms |
| **v3** | Gate knows personal-data fields, all rules, document context, no second review | dev, 30 docs | **0.990** | **0.981** | **0.964** | **0.0%** | One miss left: the judge skipped one clause–rule pair. Version frozen |
| **v3** | **Frozen version, run once** | **test, 20 docs** | **0.952** | **1.000** | **1.000** | **7.7%** | Baseline on the same set: F1 0.864, traps flagged 19.2% |