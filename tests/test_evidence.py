import pytest

from patent_reasoning.data.evidence import (
    Budget,
    PassageRecord,
    approx_tokens,
    claim_query,
    document_passages,
    normalize_document_id,
    select_passages,
)
from patent_reasoning.data.office_action import find_references
from patent_reasoning.data.publications import Publication


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("20060025037", ("20060025037", "us_publication")),
        ("US 2004/0126415 A1", ("20040126415", "us_publication")),
        ("4690653", ("4690653", "us_patent")),
        ("US 6,817,711 B2", ("6817711", "us_patent")),
        ("06817711", ("6817711", "us_patent")),
        ("D 497,245", ("D497245", "us_patent")),
        ("RE41,233 E", ("RE41233", "us_patent")),
        ("WO 03068528", ("WO03068528", "foreign")),
        ("JP 2003-109774", ("JP2003109774", "foreign")),
        ("EP 1,621,137 A1", ("EP1621137", "foreign")),
        ("Endo et al hereinafter JP 8-68587", None),
        (None, None),
    ],
)
def test_normalize_document_id(raw, expected):
    assert normalize_document_id(raw) == expected


@pytest.mark.parametrize(
    "text", ["US 2004/0126415 A1", "6,817,711 B2", "D 497,245", "WO 03/068528"]
)
def test_oard_and_text_ids_agree(text):
    """Ids parsed from the office action and from OARD normalize to the same string."""
    ((_, from_text),) = find_references(text)
    assert normalize_document_id(text)[0] == from_text


def _pub(paragraphs, abstract="An abstract."):
    return Publication("x", "A1", "Title", {}, paragraphs, abstract)


def test_document_passages_locators():
    pub = _pub({"0001": "First.", "0002": "", "0003": "Third paragraph."})
    got = [(p.passage_id, p.locator) for p in document_passages("20130173955", pub, 300)]
    assert got == [("20130173955:abstract", "Abstract"), ("20130173955:0001", "[0001]"),
                   ("20130173955:0003", "[0003]")]  # fmt: skip
    # grants don't print paragraph numbers, so their locators can't look like pin cites
    (_, para) = document_passages("7679495", _pub({"0012": "Text."}), 300)
    assert para.locator == "description paragraph 12"
    assert document_passages("WO03068528", None, 300) == []


def test_long_passages_are_truncated_to_budget():
    (p,) = document_passages("7679495", _pub({"0001": "word " * 1000}, abstract=""), 50)
    assert p.tokens <= 51 and p.text.endswith("…")


def _p(pid, text):
    return PassageRecord(pid, pid, text, approx_tokens(text))


DOCS = {
    "A": [_p("A:1", "a hinge connecting a lid to a base"), _p("A:2", "packaging and shipping"),
          _p("A:3", "a spring loaded latch holding the lid closed")],
    "B": [_p("B:1", "unrelated chemistry of polymers"), _p("B:2", "polymer solvent mixing")],
}  # fmt: skip


def test_select_passages_ranks_by_claim_and_keeps_every_document():
    query = claim_query("3. The box of claim 1, wherein the latch is spring loaded.",
                        ["1. A box comprising a hinge connecting a lid to a base."], 2)  # fmt: skip
    sel = select_passages(query, DOCS, Budget(max_tokens=1000, max_passages=3))
    # best two from A, plus B's best so every cited document is visible; reading order
    assert sel.passage_ids == ["A:1", "A:3", "B:1"] or sel.passage_ids == ["A:1", "A:3", "B:2"]
    assert sel.tokens == sum(p.tokens for ps in DOCS.values() for p in ps
                             if p.passage_id in sel.passage_ids)  # fmt: skip


def test_select_passages_respects_token_budget():
    sel = select_passages(["latch"], DOCS, Budget(max_tokens=12, max_passages=10))
    assert sel.tokens <= 12
    assert "A:3" in sel.passage_ids


def test_selection_is_deterministic():
    query = ["lid", "base"]
    assert select_passages(query, DOCS, Budget()) == select_passages(query, DOCS, Budget())


def test_cited_pool_comes_first_but_leaves_room_for_retrieval():
    budget = Budget(max_tokens=24, max_passages=10, cited_share=0.5, min_per_document=0)
    sel = select_passages(["latch"], DOCS, budget, cited={"B:1", "B:2"})
    assert "A:3" in sel.passage_ids  # the claim's best match still fits
    assert len({"B:1", "B:2"} & set(sel.passage_ids)) == 1  # the pool is capped at half


def test_each_limitation_gets_its_best_passage():
    budget = Budget(max_tokens=1000, max_passages=2, min_per_document=0, limitation_depth=1)
    queries = [["polymer", "solvent"], ["latch", "spring"]]
    sel = select_passages(["lid"], DOCS, budget, limitation_queries=queries)
    assert sel.passage_ids == ["A:3", "B:2"]  # not the two "lid" passages
