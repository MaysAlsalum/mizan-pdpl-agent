from eval.metrics import aggregate, score_doc

DOC = {
    "doc_id": "t1",
    "full_text": "1. We keep all user data forever and never delete it.\n"
                 "2. We keep invoices for the period required by tax law.\n"
                 "3. Users may not access or delete their data.\n"
                 "4. Data may be used for related purposes we decide.\n"
                 "5. Fees are payable within 30 days.",
    "clauses": [
        {"clause_id": "c1", "text": "We keep all user data forever and never delete it.", "label": "violation", "rule_id": "R06"},
        {"clause_id": "c2", "text": "We keep invoices for the period required by tax law.", "label": "trap", "rule_id": "R06"},
        {"clause_id": "c3", "text": "Users may not access or delete their data.", "label": "violation", "rule_id": "R05"},
        {"clause_id": "c4", "text": "Data may be used for related purposes we decide.", "label": "ambiguous", "rule_id": "R02"},
        {"clause_id": "c5", "text": "Fees are payable within 30 days.", "label": "neutral", "rule_id": None},
    ],
}


def out(findings, escalations=()):
    return {"findings": [{"verdict": "violation", **f} for f in findings], "escalations": list(escalations)}


def test_perfect():
    s = score_doc(DOC, out([{"rule_id": "R06", "evidence": "keep all user data forever"},
                            {"rule_id": "R05", "evidence": "Users may not access or delete their data."}]))
    assert (s["tp"], s["fp"], s["fn"]) == (2, 0, 0)
    assert aggregate([s])["f1"] == 1.0
    assert s["grounded"] == 2


def test_nothing_found():
    s = score_doc(DOC, out([]))
    assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 2)
    assert aggregate([s])["f1"] == 0.0


def test_trap_is_false_positive():
    s = score_doc(DOC, out([{"rule_id": "R06", "evidence": "keep invoices for the period required by tax law"}]))
    assert (s["tp"], s["fp"], s["trap_hits"]) == (0, 1, 1)
    assert aggregate([s])["trap_fp_rate"] == 1.0


def test_right_rule_wrong_clause_is_wrong():
    s = score_doc(DOC, out([{"rule_id": "R05", "evidence": "Fees are payable within 30 days."}]))
    assert (s["tp"], s["fp"]) == (0, 1)


def test_ambiguous_flag_is_neutral_and_escalation_counts():
    s = score_doc(DOC, out([{"rule_id": "R02", "evidence": "used for related purposes we decide"}],
                           [{"rule_id": "R02", "evidence": "used for related purposes we decide"}]))
    assert (s["tp"], s["fp"]) == (0, 0)
    assert s["ambiguous_escalated"] == 1


def test_second_rule_on_violating_clause_is_not_penalised():
    s = score_doc(DOC, out([{"rule_id": "R06", "evidence": "keep all user data forever"},
                            {"rule_id": "R02", "evidence": "keep all user data forever"}]))
    assert (s["tp"], s["fp"], s["unverified"]) == (1, 0, 1)


def test_wrong_rule_on_violating_clause_still_misses():
    s = score_doc(DOC, out([{"rule_id": "R02", "evidence": "keep all user data forever"}]))
    assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 2)


def test_invented_evidence_not_grounded():
    s = score_doc(DOC, out([{"rule_id": "R06", "evidence": "The vendor sells data to advertisers worldwide."}]))
    assert s["grounded"] == 0