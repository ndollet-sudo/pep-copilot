from pepcopilot.chunking import (chunk_doc, fixed_windows, merge_sections, pack_section,
                                 split_sections)


def test_fixed_windows_deck_case():
    toy = " ".join(f"w{i}" for i in range(1000))
    chunks = fixed_windows(toy, c=200, o=50)
    assert len(chunks) == 7                                # ceil(1000 / 150)
    assert len(chunks[0].split()) == 200
    assert len(chunks[-1].split()) == 100
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]


def test_fixed_windows_no_redundant_tail():
    # 950 words: the window starting at 750 already reaches the end -> no 7th fully-contained chunk
    chunks = fixed_windows(" ".join(f"w{i}" for i in range(950)), 200, 50)
    assert chunks[-1].split()[-1] == "w949" and len(chunks) == 6


RST = """Preamble text here.

Abstract
========

This is the abstract paragraph.

Rationale
---------

First paragraph of rationale.

Second paragraph of rationale.

Copyright
=========

This document is placed in the public domain.
"""


def test_split_sections_finds_titles():
    titles = [t for t, _ in split_sections(RST)]
    assert titles == ["Preamble", "Abstract", "Rationale", "Copyright"]


def test_merge_drops_boilerplate_and_glues_short_sections():
    merged = merge_sections(split_sections(RST), min_words=10)
    assert all("Copyright" not in t for t, _ in merged)
    assert len(merged) == 1 and "Abstract" in merged[0][0]


def test_pack_section_respects_budget_and_overlap():
    body = "\n\n".join(" ".join(f"p{j}w{i}" for i in range(80)) for j in range(5))
    chunks = pack_section(body, max_words=200, overlap=20)
    assert len(chunks) >= 2
    assert all(len(c.split()) <= 220 for c in chunks)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_chunk_doc_carries_metadata_and_header():
    doc = {"doc_id": "pep-9000", "pep": 9000, "title": "Toy", "status": "Final",
           "python_version": "3.99", "text": RST}
    for strategy in ("fixed", "fixed_header", "section"):
        chunks = chunk_doc(doc, strategy)
        assert chunks and all(c["doc_id"] == "pep-9000" for c in chunks)
        assert chunks[0]["chunk_id"] == "pep-9000#c0"
    assert chunk_doc(doc, "section")[0]["embed_text"].startswith("PEP 9000: Toy [Final, Python 3.99]")
    assert not chunk_doc(doc, "fixed")[0]["embed_text"].startswith("PEP")
