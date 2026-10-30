from src.agent import candidate_rules, guess_doc_type, segment
from src.catalog import load_rules


def test_segment_english_numbers():
    text = "Title\n\n1. First clause.\n\n2. Second clause.\n\n3. Third clause."
    assert [c["text"] for c in segment(text)] == ["First clause.", "Second clause.", "Third clause."]


def test_segment_arabic_articles_and_digits():
    text = "عنوان\n\nالمادة (١): البند الأول.\n\nالمادة (٢): البند الثاني."
    assert [c["text"] for c in segment(text)] == ["البند الأول.", "البند الثاني."]


def test_segment_ignores_out_of_sequence_numbers_and_page_marks():
    text = ("Title\n\n(1) Notice must be given\n(60) days before.\n- 2 -\n\n(2) Fees are due\nmonthly.")
    clauses = [c["text"] for c in segment(text)]
    assert clauses == ["Notice must be given (60) days before.", "Fees are due monthly."]


def test_retrieval_uses_keywords():
    rules = load_rules()
    picked = [r["id"] for r in candidate_rules("We retain data indefinitely and never delete it.",
                                               rules, "privacy_policy", True)]
    assert "R06" in picked


def test_neutral_clause_gets_no_rules():
    assert candidate_rules("Fees are payable within 30 days.", load_rules(), "vendor_contract", True) == []


def test_doc_type_guess():
    assert guess_doc_type("Services Agreement between the Company and the Vendor") == "vendor_contract"