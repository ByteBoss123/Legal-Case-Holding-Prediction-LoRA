"""Unit tests on hand-written examples (format/logic checks only; not CaseHOLD data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lh  # noqa: E402


def test_tort_filter():
    assert lh.is_tort("The plaintiff alleged negligence after the slip and fall.")
    assert lh.is_tort("a wrongful death action")
    assert not lh.is_tort("The contract was ambiguous as to price escalation.")
    assert not lh.is_tort("the retortion doctrine")  # word boundary
    assert not lh.is_tort("claims under the Convention Against Torture")
    assert lh.is_tort("tortious interference by the tortfeasor")


def test_prompt_and_pick():
    p = lh.mc_prompt("ctx (<HOLDING>)", ["h0", "h1", "h2", "h3", "h4"])
    assert "A. h0" in p and "E. h4" in p and "<HOLDING>" in p
    assert lh.pick([0.1, 0.5, 0.5, 0.2, 0.0]) == 1


def test_tfidf_baseline():
    train = ["holding that a landowner owes a duty of care to invitees",
             "holding that the statute of limitations bars the claim",
             "holding that the contract was unenforceable"] * 3
    ctx = ["The landowner breached the duty of care owed to the invitee (<HOLDING>)."]
    ends = [["holding that the contract was unenforceable",
             "holding that a landowner owes a duty of care to invitees",
             "holding that the statute of limitations bars the claim",
             "holding that the contract was unenforceable", "holding that the contract was unenforceable"]]
    assert lh.tfidf_predict(train, ctx, ends) == [1]


def test_metrics_and_bootstrap():
    gold = [0, 1, 2, 3, 4] * 20
    m = lh.metrics(gold, gold)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    allA = [0] * len(gold)
    assert abs(lh.metrics(gold, allA)["accuracy"] - 0.2) < 1e-9
    assert lh.metrics(gold, allA)["pred_letter_share"]["A"] == 1.0
    b = lh.paired_bootstrap(gold, gold, allA, n=300)
    assert abs(b["diff"] - 0.8) < 1e-9 and b["ci95"][0] > 0.6
    sub = lh.accuracy_ci(gold, allA, idx=[i for i, g in enumerate(gold) if g == 0], n=200)
    assert sub["accuracy"] == 1.0 and sub["n"] == 20
