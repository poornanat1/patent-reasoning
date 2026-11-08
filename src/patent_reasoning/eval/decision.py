"""Decision metrics over the five classes {no rejection, 102, 103, 112, 101}.

A claim can have several grounds, so each class is scored as its own yes/no question: the
example is a true positive for 103 if both the target and the prediction include 103.
"No rejection" is the class of an empty set of grounds. Macro-F1 averages the per-class F1
over the classes present in the gold labels.

The decision is read from the output's grounds even when the output breaks the
evidence-status rule (generation guarantees the grounds themselves are well-formed);
validity is reported separately. An output that isn't JSON predicts nothing: a false
negative for every gold class, and a false positive for none.

Confidence intervals resample whole applications, not claims: claims in one office action
share prior art and an examiner, so they aren't independent.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence

from patent_reasoning.schema import DECISION_CLASSES, NO_REJECTION

Decision = frozenset[str]
STATUTES = frozenset(DECISION_CLASSES) - {NO_REJECTION}


def predicted(text: str) -> Decision | None:
    """The decision in a raw output, or None if it can't be read."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    grounds = data.get("grounds") if isinstance(data, dict) else None
    if not isinstance(grounds, list):
        return None
    statutes = frozenset(g.get("statute") for g in grounds if isinstance(g, dict)) & STATUTES
    return statutes or frozenset({NO_REJECTION})


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(gold: Sequence[Decision], pred: Sequence[Decision | None]) -> float:
    return metrics(gold, pred)["macro_f1"]


def metrics(gold: Sequence[Decision], pred: Sequence[Decision | None]) -> dict:
    per_class = {}
    for c in DECISION_CLASSES:
        tp = sum(c in g and p is not None and c in p for g, p in zip(gold, pred, strict=True))
        fp = sum(c not in g and p is not None and c in p for g, p in zip(gold, pred, strict=True))
        fn = sum(c in g and (p is None or c not in p) for g, p in zip(gold, pred, strict=True))
        precision, recall, f1 = _f1(tp, fp, fn)
        per_class[c] = {"support": tp + fn, "predicted": tp + fp, "precision": round(precision, 4),
                        "recall": round(recall, 4), "f1": round(f1, 4)}  # fmt: skip
    present = [c for c in DECISION_CLASSES if per_class[c]["support"]]
    allowable = [p for g, p in zip(gold, pred, strict=True) if NO_REJECTION in g]
    rejected = [p for g, p in zip(gold, pred, strict=True) if NO_REJECTION not in g]
    n = len(gold)
    return {
        "examples": n,
        "macro_f1": round(sum(per_class[c]["f1"] for c in present) / len(present), 4)
        if present
        else 0.0,
        "exact_match": round(sum(g == p for g, p in zip(gold, pred, strict=True)) / n, 4),
        # Rejecting a claim the examiner found allowable, and the reverse.
        "false_rejection_rate": round(
            sum(p is not None and NO_REJECTION not in p for p in allowable) / len(allowable), 4
        )
        if allowable
        else None,
        "false_allowance_rate": round(
            sum(p is not None and NO_REJECTION in p for p in rejected) / len(rejected), 4
        )
        if rejected
        else None,
        "unreadable_rate": round(sum(p is None for p in pred) / n, 4),
        "per_class": per_class,
    }


def _resample(clusters: Sequence[str], rng: random.Random) -> list[int]:
    by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    groups = list(by_cluster.values())
    return [i for _ in groups for i in rng.choice(groups)]


def bootstrap(
    gold: Sequence[Decision],
    pred: Sequence[Decision | None],
    clusters: Sequence[str],
    stat: Callable = macro_f1,
    n: int = 1000,
    seed: int = 0,
) -> list[float]:
    """95% interval of ``stat``, resampling ``clusters`` (application IDs) with replacement."""
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        idx = _resample(clusters, rng)
        values.append(stat([gold[i] for i in idx], [pred[i] for i in idx]))
    values.sort()
    return [round(values[int(0.025 * n)], 4), round(values[int(0.975 * n) - 1], 4)]


def paired_bootstrap(
    gold: Sequence[Decision],
    pred_a: Sequence[Decision | None],
    pred_b: Sequence[Decision | None],
    clusters: Sequence[str],
    stat: Callable = macro_f1,
    n: int = 1000,
    seed: int = 0,
) -> dict:
    """``stat(b) - stat(a)`` on the same resamples, with its 95% interval and the share of
    resamples where b is better."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        idx = _resample(clusters, rng)
        g = [gold[i] for i in idx]
        diffs.append(stat(g, [pred_b[i] for i in idx]) - stat(g, [pred_a[i] for i in idx]))
    diffs.sort()
    return {
        "difference": round(stat(gold, pred_b) - stat(gold, pred_a), 4),
        "ci95": [round(diffs[int(0.025 * n)], 4), round(diffs[int(0.975 * n) - 1], 4)],
        "b_better_share": round(sum(d > 0 for d in diffs) / n, 4),
    }
