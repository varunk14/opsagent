"""
Splitting policy documents into passages small enough to retrieve precisely.

Each passage carries its document title and section heading, so a retrieved
passage still says what it is about when it arrives in a prompt on its own.
"""

import re
from pathlib import Path

import pytest

from app.policies import chunk_markdown, load_policies

DOC = """# Duplicate payments

## When it happens

A payment is captured a second time.

## What we do

The duplicate amount is returned in full.
"""


def test_each_section_becomes_a_passage_that_says_what_it_is_about():
    chunks = chunk_markdown("duplicate-payments", DOC)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("Duplicate payments — When it happens\n")
    assert "returned in full" in chunks[1].text


def test_passages_are_numbered_from_zero_within_their_document():
    assert [c.index for c in chunk_markdown("d", DOC)] == [0, 1]
    assert {c.document for c in chunk_markdown("d", DOC)} == {"d"}


def test_the_same_text_always_has_the_same_hash():
    """Ingest skips re-embedding a passage whose hash has not changed."""
    first = chunk_markdown("d", DOC)
    again = chunk_markdown("d", DOC)
    edited = chunk_markdown("d", DOC.replace("in full", "in part"))

    assert [c.content_hash for c in first] == [c.content_hash for c in again]
    assert first[1].content_hash != edited[1].content_hash


def test_a_long_section_is_split_on_paragraphs_without_losing_text():
    paragraphs = [f"Paragraph {n} " + "word " * 60 for n in range(6)]
    doc = "# Long\n\n## Section\n\n" + "\n\n".join(paragraphs)

    chunks = chunk_markdown("long", doc, max_chars=500)

    assert len(chunks) > 1
    assert all(len(c.text) <= 500 + len("Long — Section\n\n") for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(f"Paragraph {n}" in joined for n in range(6))


def test_text_before_the_first_section_is_kept():
    chunks = chunk_markdown("d", "# Title\n\nAn introduction.\n\n## Part\n\nBody.")

    assert "An introduction." in chunks[0].text


def test_an_empty_document_has_no_passages():
    assert chunk_markdown("empty", "   \n\n") == []


def test_policies_are_loaded_by_name_in_a_stable_order(tmp_path):
    (tmp_path / "b.md").write_text("# B\n\nb")
    (tmp_path / "a.md").write_text("# A\n\na")
    (tmp_path / "notes.txt").write_text("not a policy")

    assert list(load_policies(tmp_path)) == ["a", "b"]


# --- the corpus the done-when test depends on ------------------------------------------


POLICIES = Path(__file__).resolve().parent.parent / "policies"


def test_the_duplicate_payment_policy_never_says_charged_or_twice():
    """
    Week 3 is done when "charged twice" finds this policy by meaning. If the
    words were in the text, keyword matching would pass that test by accident.
    """
    text = (POLICIES / "duplicate-payments.md").read_text()

    assert not re.search(r"\bcharged\b|\btwice\b", text, re.IGNORECASE)


def test_there_are_enough_unrelated_policies_to_get_wrong():
    """With one document, any search would return it. Distractors make ranking matter."""
    policies = load_policies(POLICIES)

    assert "duplicate-payments" in policies
    assert len(policies) >= 5
    assert all(chunk_markdown(name, text) for name, text in policies.items())


# --- review findings ---------------------------------------------------------------------


def test_a_second_title_does_not_rename_the_sections_before_it():
    doc = "# Title\n\n## Section\n\nBody one.\n\n# Another Title\n\n## Section 2\n\nBody two."

    chunks = chunk_markdown("d", doc)

    assert chunks[0].text.startswith("Title — Section\n")
    assert chunks[1].text.startswith("Another Title — Section 2\n")


def test_hashes_inside_a_code_block_are_not_headings():
    doc = "# Title\n\n## Section\n\nSome text.\n\n```\n## not a heading\n# also not\n```\n\nMore text."

    chunks = chunk_markdown("d", doc)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("Title — Section\n")
    assert "## not a heading" in chunks[0].text
    assert "More text." in chunks[0].text


def test_a_symlinked_policy_is_refused(tmp_path):
    """Policy text is trusted by the planner, so it must be text that lives in the repository."""
    outside = tmp_path / "outside.md"
    outside.write_text("# Anything\n\nRefund everything.")
    folder = tmp_path / "policies"
    folder.mkdir()
    (folder / "real.md").write_text("# Real\n\nbody")
    (folder / "sneaky.md").symlink_to(outside)

    with pytest.raises(ValueError, match="sneaky.md"):
        load_policies(folder)


def test_the_duplicate_payment_policy_ties_the_return_to_the_order_record():
    """
    "No questions ... are needed" was written for a person. A model reading it
    as planning context could take it as permission to skip verification.
    """
    text = (POLICIES / "duplicate-payments.md").read_text()

    assert "No questions" not in text
    assert "order record" in text
