"""Build the 500-document knowledge base from the official python/peps repository.

Usage:
    python scripts/build_corpus.py                 # clones python/peps into data/raw/
    python scripts/build_corpus.py --src /path/to/peps/peps

Selection rule (deterministic, documented in the README):
  1. drop "April Fool!" PEPs and PEPs with fewer than 300 words;
  2. rank by status priority: Final > Active > Accepted > Provisional > Superseded
     > Rejected > Withdrawn > Deferred > Draft, ties broken by PEP number;
  3. keep the first 500.
Rejected and Superseded PEPs are kept on purpose: they are "hard negatives"
(e.g. PEP 622 was superseded by PEP 634) that a retriever must not confuse.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus" / "peps.jsonl"
N_DOCS = 500
MIN_WORDS = 300
STATUS_PRIORITY = ["Final", "Active", "Accepted", "Provisional", "Superseded",
                   "Rejected", "Withdrawn", "Deferred", "Draft"]
HEADER_FIELDS = ["PEP", "Title", "Author", "Status", "Type", "Created",
                 "Python-Version", "Replaces", "Superseded-By", "Requires"]


def parse_header(raw: str) -> tuple[dict, str]:
    """RFC-822 style header (with continuation lines) followed by the body."""
    header_block, _, body = raw.partition("\n\n")
    fields, key = {}, None
    for line in header_block.splitlines():
        if line[:1].isspace() and key:
            fields[key] += " " + line.strip()
        elif ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            fields[key] = val.strip()
    return fields, body


def clean_rst(body: str) -> str:
    """Light RST cleanup: keep the prose and the section titles, drop markup noise."""
    out = []
    for line in body.splitlines():
        s = line.rstrip()
        if re.match(r"^\.\. (_[^:]+:|canonical-doc::|canonical-pypa-spec::|highlight::|"
                     r"include::|toctree::|contents::|image::|figure::)", s.strip()):
            continue
        if re.match(r"^\s*:(maxdepth|local|depth|target|alt|width|class):", s):
            continue
        s = re.sub(r":(?:pep|ref|rfc|func|class|mod|meth|attr|data|exc|const|term|"
                   r"external\+\w+:\w+|c:\w+|py:\w+|token|keyword|envvar|file|program|"
                   r"option|samp)?:`~?([^`<]*?)(?:\s*<[^>]+>)?`", r"\1", s)
        s = s.replace("``", "")
        s = re.sub(r"`([^`<]+?)\s*<[^>]+>`_+", r"\1", s)
        out.append(s)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_emails(authors: str) -> str:
    return re.sub(r"\s*<[^>]+>", "", authors).strip()


def load_all(src: str) -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(src, "pep-*.rst"))):
        raw = open(path, encoding="utf-8").read()
        fields, body = parse_header(raw)
        if "PEP" not in fields:
            continue
        text = clean_rst(body)
        n = int(fields["PEP"])
        docs.append({
            "doc_id": f"pep-{n:04d}",
            "pep": n,
            "title": fields.get("Title", ""),
            "authors": strip_emails(fields.get("Author", "")),
            "status": fields.get("Status", ""),
            "type": fields.get("Type", ""),
            "created": fields.get("Created", ""),
            "python_version": fields.get("Python-Version", ""),
            "replaces": fields.get("Replaces", ""),
            "superseded_by": fields.get("Superseded-By", ""),
            "url": f"https://peps.python.org/pep-{n:04d}/",
            "n_words": len(text.split()),
            "text": text,
        })
    return docs


def select(docs: list[dict], n: int = N_DOCS) -> list[dict]:
    pool = [d for d in docs if d["status"] != "April Fool!" and d["n_words"] >= MIN_WORDS]
    rank = {s: i for i, s in enumerate(STATUS_PRIORITY)}
    pool.sort(key=lambda d: (rank.get(d["status"], 99), d["pep"]))
    chosen = pool[:n]
    chosen.sort(key=lambda d: d["pep"])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="folder containing pep-XXXX.rst files")
    args = ap.parse_args()
    src = args.src
    if src is None:
        raw_dir = ROOT / "data" / "raw" / "peps"
        if not raw_dir.exists():
            raw_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/python/peps", str(raw_dir)], check=True)
        src = str(raw_dir / "peps")
    docs = select(load_all(src))
    assert len(docs) == N_DOCS, f"expected {N_DOCS} documents, got {len(docs)}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(docs)} documents -> {OUT.relative_to(ROOT)}")
    print("status:", dict(Counter(d["status"] for d in docs)))
    print("type:  ", dict(Counter(d["type"] for d in docs)))
    print(f"words: {sum(d['n_words'] for d in docs):,}")


if __name__ == "__main__":
    main()
