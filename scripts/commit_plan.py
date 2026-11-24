"""Team commit plan: one commit per member (A first, then B, C, D in any order).

    python scripts/commit_plan.py              # full plan
    python scripts/commit_plan.py --member B   # only B's steps, as git commands
    python scripts/commit_plan.py --check      # every project file is assigned exactly once

Each member commits from their OWN machine with their OWN git identity
(git config user.name / user.email), after `git pull --rebase`, in step order.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MEMBERS = {
    "A": "Data & Evaluation lead",
    "B": "Retrieval lead",
    "C": "LLM, generation & instrumentation lead",
    "D": "Agent lead",
}

PLAN = [
    ('A', 'feat(data+eval): 500-PEP corpus, golden sets, retrieval metrics, README',
     ['.gitignore', 'requirements.txt', '.env.example', 'Makefile', 'src/pepcopilot/__init__.py', 'src/pepcopilot/config.py', 'scripts/commit_plan.py', 'docs/WORK_SPLIT.md', 'scripts/build_corpus.py', 'src/pepcopilot/corpus.py', 'data/corpus/peps.jsonl', 'data/golden/retrieval_dev.jsonl', 'data/golden/retrieval_test.jsonl', 'src/pepcopilot/eval/__init__.py', 'src/pepcopilot/eval/metrics.py', 'src/pepcopilot/eval/retrieval_eval.py', 'tests/test_retrieval_and_metrics.py', 'README.md', 'docs/DEFENCE_PREP.md']),
    ('B', 'feat(retrieval): chunking strategies, BM25 + FAISS hybrid with RRF, reranker, scorecards',
     ['src/pepcopilot/chunking.py', 'tests/test_chunking.py', 'src/pepcopilot/embeddings.py', 'src/pepcopilot/retrieval.py', 'src/pepcopilot/rerank.py', 'results/retrieval_dev_lsa.csv', 'results/retrieval_dev_lsa.md', 'results/retrieval_dev_lsa_per_question.csv', 'results/retrieval_test_lsa.csv', 'results/retrieval_test_lsa.md', 'results/retrieval_test_lsa_per_question.csv']),
    ('C', 'feat(llm): OpenAI client with backoff, telemetry, cited generation, RAG pipeline, e2e eval',
     ['src/pepcopilot/llm.py', 'src/pepcopilot/telemetry.py', 'src/pepcopilot/generate.py', 'src/pepcopilot/pipeline.py', 'src/pepcopilot/__main__.py', 'src/pepcopilot/eval/agent_eval.py', 'src/pepcopilot/eval/faithfulness.py', 'data/golden/agent_test.jsonl', 'tests/test_eval_checks.py']),
    ('D', 'feat(agent): tool registry, multi-step loop with failure handling, LangGraph port, architecture',
     ['src/pepcopilot/agent/__init__.py', 'src/pepcopilot/agent/tools.py', 'tests/conftest.py', 'tests/test_tools.py', 'src/pepcopilot/agent/loop.py', 'src/pepcopilot/agent/graph.py', 'tests/test_agent_failures.py', 'docs/architecture.md']),
]

# Optional, after the paid runs (needs OPENAI_API_KEY): whoever runs them commits the results.
AFTER_API_RUNS = [
    ("B", "make eval-retrieval", "results: API-embedder + rerank retrieval scorecards",
     ["results/retrieval_dev_api*", "results/retrieval_test_api*"]),
    ("C", "make eval-agent", "results: agent vs RAG-only held-out scorecard",
     ["results/agent_test.md", "results/agent_test_runs.csv", "results/rag_test_runs.csv"]),
]

IGNORED_PREFIXES = ("data/raw/", "data/index/", "results/traces/", ".git/", ".pytest_cache/")


def project_files() -> set[str]:
    out = set()
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and "__pycache__" not in rel and not rel.startswith(IGNORED_PREFIXES) \
                and rel != ".env":
            out.add(rel)
    return out


def check() -> int:
    planned = [f for _, _, files in PLAN for f in files]
    dupes = {f for f in planned if planned.count(f) > 1}
    missing = project_files() - set(planned)
    absent = set(planned) - project_files()
    for label, items in (("assigned twice", dupes), ("not assigned", missing), ("planned but absent", absent)):
        for f in sorted(items):
            print(f"{label}: {f}")
    ok = not (dupes or missing or absent)
    print("plan OK: every file is assigned exactly once" if ok else "plan has problems")
    return 0 if ok else 1


def show(member: str | None) -> None:
    for i, (m, msg, files) in enumerate(PLAN, start=1):
        if member and m != member:
            continue
        print(f"\n# step {i:02d} - member {m} ({MEMBERS[m]})")
        print("git pull --rebase")
        print("git add " + " ".join(files))
        print(f'git commit -m "{msg}"')
        print("git push")
    print("\n# --- after the API runs (needs OPENAI_API_KEY) ---")
    for m, cmd, msg, files in AFTER_API_RUNS:
        if member and m != member:
            continue
        print(f"\n# member {m}: {cmd}\ngit pull --rebase\ngit add {' '.join(files)}\n"
              f'git commit -m "{msg}"\ngit push')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", choices=sorted(MEMBERS))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(check()) if a.check else show(a.member)
