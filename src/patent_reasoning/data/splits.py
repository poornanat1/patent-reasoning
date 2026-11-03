"""Train / validation / test splits that keep application families disjoint.

- Out-of-distribution test: every family with an office action examined in one of the
  ``ood_technology_centers`` (the first two digits of the art unit, e.g. "16" for
  biotechnology), so that technology is unseen in training.
- The other families go to train, validation or in-distribution test by a salted hash of
  the family ID, so the assignment doesn't depend on row order or on which other
  examples exist.
- Validation and both test sets keep only fully grounded examples (see
  ``fully_grounded``). The less certain examples of those families are dropped, not
  moved to train, which would leak the family.
"""

from __future__ import annotations

import hashlib

from patent_reasoning.schema import Example

SPLITS = ("train", "val", "test", "ood_test")
EVAL_SPLITS = ("val", "test", "ood_test")


def fully_grounded(example: Example) -> bool:
    """Claim text from the publication (not OCR), and a target that isn't partial guesswork:
    no unknown or partial mapping for the decision, and an explanation to score against
    whenever the claim is rejected."""
    flags = set(example.provenance.flags)
    if example.provenance.claims_source != "publication":
        return False
    if flags & {"mapping_partial", "restatement_low_overlap"}:
        return False
    return not (example.target.grounds and example.target.explanation is None)


def family_split(family: str, salt: str, val: float, test: float) -> str:
    u = int(hashlib.sha256(f"{salt}:{family}".encode()).hexdigest()[:12], 16) / 16**12
    if u < val:
        return "val"
    if u < val + test:
        return "test"
    return "train"


def assign(
    examples: list[Example],
    family_of: dict[str, str],
    salt: str,
    val: float,
    test: float,
    ood_technology_centers: list[str],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Returns (family -> split, example ID -> split, example IDs dropped from eval splits)."""
    ood = {family_of[e.provenance.application_id] for e in examples
           if e.provenance.art_unit[:2] in ood_technology_centers}  # fmt: skip
    split_of_family = {}
    for family in sorted(set(family_of[e.provenance.application_id] for e in examples)):
        split_of_family[family] = (
            "ood_test" if family in ood else family_split(family, salt, val, test)
        )
    split_of_example, dropped = {}, []
    for e in examples:
        split = split_of_family[family_of[e.provenance.application_id]]
        if split in EVAL_SPLITS and not fully_grounded(e):
            dropped.append(e.example_id)
            continue
        split_of_example[e.example_id] = split
    return split_of_family, split_of_example, dropped
