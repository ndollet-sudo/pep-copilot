"""Chunking strategies. Granularity is a design choice - we compare three of them.

1. ``fixed``          : 200-word windows, 50-word overlap (the session-5 baseline).
                        n = ceil(L / (c - o)) chunks for a document of L words.
2. ``fixed_header``   : same windows, each prefixed by a *contextual header*
                        ("PEP 572: Assignment Expressions [Final, Python 3.8] > Rationale").
3. ``section``        : structure-aware - split on the RST section titles, drop boilerplate
                        sections (Copyright, References...), merge tiny sections, pack whole
                        paragraphs up to ~220 words with a 40-word overlap inside a section,
                        plus the contextual header. A chunk never cuts a paragraph mid-way
                        unless the paragraph alone exceeds the budget.

Every chunk keeps its document metadata so that citations can name the PEP.
"""
from __future__ import annotations

import re
from typing import Iterable

from . import config

_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


def fixed_windows(text: str, c: int = config.FIXED_CHUNK_WORDS,
                  o: int = config.FIXED_OVERLAP_WORDS) -> list[str]:
    """Chunk i covers words[i*(c-o) : i*(c-o)+c]; the last window may be short."""
    if not 0 <= o < c:
        raise ValueError("overlap must satisfy 0 <= o < c")
    words = text.split()
    step, chunks, start = c - o, [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + c]))
        if start + c >= len(words):
            break
        start += step
    return chunks


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split RST text into (section title, body) using underlined titles.
    Text before the first title goes into a section called 'Preamble'."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [("Preamble", [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        is_title = (line.strip() and not _UNDERLINE.match(line) and _UNDERLINE.match(nxt)
                    and len(nxt.strip()) >= max(3, len(line.strip()) - 2))
        if is_title:
            sections.append((line.strip(), []))
            i += 2
            continue
        if _UNDERLINE.match(line):  # overline of an over+underlined title, or a rule
            i += 1
            continue
        sections[-1][1].append(line)
        i += 1
    return [(t, "\n".join(b).strip()) for t, b in sections if "\n".join(b).strip()]


def _paragraphs(body: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def pack_section(body: str, max_words: int = config.SECTION_MAX_WORDS,
                 overlap: int = config.SECTION_OVERLAP_WORDS) -> list[str]:
    """Greedy paragraph packing; oversize paragraphs fall back to fixed windows.
    Consecutive chunks of a section share the last ``overlap`` words."""
    units: list[str] = []
    for p in _paragraphs(body):
        n = len(p.split())
        units.extend(fixed_windows(p, max_words, overlap) if n > max_words else [p])
    chunks: list[str] = []
    cur: list[str] = []
    for u in units:
        if cur and len(" ".join(cur + [u]).split()) > max_words:
            chunks.append(" ".join(cur))
            tail = " ".join(" ".join(cur).split()[-overlap:]) if overlap else ""
            cur = [tail] if tail else []
        cur.append(u)
    if cur:
        chunks.append(" ".join(cur))
    return [re.sub(r"\s+", " ", c).strip() for c in chunks if c.strip()]


_BOILERPLATE = re.compile(r"^(copyright|references?|footnotes?|acknowledge?ments?|"
                          r"change ?history|version history|resolution)$", re.I)


def merge_sections(sections: list[tuple[str, str]], min_words: int = 60) -> list[tuple[str, str]]:
    """Drop boilerplate sections (Copyright, References, ...) and glue sections shorter than
    ``min_words`` onto the next one, so tiny chunks do not flood the index."""
    kept = [(t, b) for t, b in sections if not _BOILERPLATE.match(t.strip())]
    out: list[tuple[str, str]] = []
    pending_t: list[str] = []
    pending_b: list[str] = []
    for t, b in kept:
        pending_t.append(t)
        pending_b.append(b)
        if len(" ".join(pending_b).split()) >= min_words:
            out.append((" / ".join(pending_t), "\n\n".join(pending_b)))
            pending_t, pending_b = [], []
    if pending_b:
        if out:
            t, b = out.pop()
            out.append((t + " / " + " / ".join(pending_t), b + "\n\n" + "\n\n".join(pending_b)))
        else:
            out.append((" / ".join(pending_t), "\n\n".join(pending_b)))
    return out


def header(doc: dict, section: str | None = None) -> str:
    meta = [doc.get("status", "")]
    if doc.get("python_version"):
        meta.append(f"Python {doc['python_version']}")
    h = f"PEP {doc['pep']}: {doc['title']} [{', '.join(m for m in meta if m)}]"
    return f"{h} > {section}" if section else h


def chunk_doc(doc: dict, strategy: str = "section") -> list[dict]:
    if strategy in ("fixed", "fixed_header"):
        pieces = [(None, t) for t in fixed_windows(doc["text"])]
    elif strategy == "section":
        pieces = [(title, t) for title, body in merge_sections(split_sections(doc["text"]))
                  for t in pack_section(body)]
    else:
        raise ValueError(f"unknown chunking strategy: {strategy}")
    chunks = []
    for j, (section, text) in enumerate(pieces):
        h = header(doc, section) if strategy != "fixed" else ""
        chunks.append({
            "chunk_id": f"{doc['doc_id']}#c{j}",
            "doc_id": doc["doc_id"], "pep": doc["pep"], "title": doc["title"],
            "status": doc["status"], "section": section or "", "n": j,
            "text": text,
            "embed_text": f"{h}\n{text}" if h else text,
        })
    return chunks


def chunk_corpus(docs: Iterable[dict], strategy: str = "section") -> list[dict]:
    return [c for d in docs for c in chunk_doc(d, strategy)]
