import sys

sys.path.insert(0, "scripts")

from tune_thresholds import ORDER, decide, fit  # noqa: E402

from patent_reasoning.eval.decision import macro_f1  # noqa: E402


def row(gold: set[str], **score: float) -> dict:
    full = dict.fromkeys(ORDER, 0.0)
    full.update({k.removeprefix("s"): v for k, v in score.items()})
    return {"app": "a", "gold": frozenset(gold), "greedy": None, "score": full, "n": 4}


def test_decide_predicts_every_class_at_or_above_its_threshold():
    t = dict.fromkeys(ORDER, 0.5) | {"101": 0.25}
    r = row({"102"}, s102=0.75, s101=0.25, s103=0.25)
    assert decide(r, t) == frozenset({"102", "101"})


def test_decide_falls_back_to_no_rejection():
    assert decide(row({"no_rejection"}, s103=0.25), dict.fromkeys(ORDER, 0.5)) == frozenset(
        {"no_rejection"}
    )


def test_fit_lowers_the_threshold_of_a_class_greedy_never_predicts():
    # 101 is right whenever a quarter of samples say it, and never otherwise.
    rows = [row({"102", "101"}, s102=1.0, s101=0.25) for _ in range(5)]
    rows += [row({"102"}, s102=1.0) for _ in range(5)]
    t = fit(rows)
    assert t["101"] <= 0.25
    gold = [r["gold"] for r in rows]
    assert macro_f1(gold, [decide(r, t) for r in rows]) == 1.0
