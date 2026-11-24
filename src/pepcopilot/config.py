"""Central configuration: paths, models, prices, knobs. One place to change a setting."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS_PATH = DATA / "corpus" / "peps.jsonl"
GOLDEN_DIR = DATA / "golden"
INDEX_DIR = DATA / "index"          # FAISS indexes + embedding cache (git-ignored)
RESULTS_DIR = ROOT / "results"
TRACES_DIR = RESULTS_DIR / "traces"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, never overrides the shell."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# --- LLM provider: OpenAI directly, or OpenRouter (Mistral, as in the labs). Same SDK. ---
PROVIDERS = {
    "openai": {
        "key_env": "OPENAI_API_KEY", "base_url": None,
        "chat_model": "gpt-4o-mini", "embed_model": "text-embedding-3-small",
        # USD per 1M tokens - check https://openai.com/api/pricing before reporting costs
        "prices": {"chat_in": 0.15, "chat_out": 0.60, "embed_in": 0.02},
        "extra_body": None,
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
        "chat_model": "mistralai/mistral-small-2603", "embed_model": "mistralai/mistral-embed-2312",
        "prices": {"chat_in": 0.15, "chat_out": 0.60, "embed_in": 0.10},
        "extra_body": {"provider": {"order": ["Mistral"]}},   # pin provider: honest meter
    },
}


def _pick_provider() -> str:
    explicit = os.environ.get("PEPCOPILOT_PROVIDER")
    if explicit:
        return explicit
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "openrouter"


PROVIDER = _pick_provider()
if PROVIDER not in PROVIDERS:
    raise ValueError(f"PEPCOPILOT_PROVIDER must be one of {list(PROVIDERS)}")
_P = PROVIDERS[PROVIDER]
API_KEY_ENV = _P["key_env"]
BASE_URL = _P["base_url"]
CHAT_MODEL = os.environ.get("PEPCOPILOT_CHAT_MODEL", _P["chat_model"])
EMBED_MODEL = os.environ.get("PEPCOPILOT_EMBED_MODEL", _P["embed_model"])
PRICES = {k: float(os.environ.get(f"PEPCOPILOT_PRICE_{k.upper()}", v))
          for k, v in _P["prices"].items()}
EXTRA_BODY = _P["extra_body"]

# --- Retry policy (exponential backoff, cf. session 4/6) ---
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 2.0
PERMANENT_HTTP_ERRORS = (400, 401, 402, 403, 404, 422)

# --- Chunking ---
# Chosen on the DEV split (results/retrieval_dev_*.md); re-check with the API embedder.
DEFAULT_STRATEGY = os.environ.get("PEPCOPILOT_CHUNKING", "fixed_header")
FIXED_CHUNK_WORDS = 200
FIXED_OVERLAP_WORDS = 50
SECTION_MAX_WORDS = 220
SECTION_OVERLAP_WORDS = 40

# --- Retrieval ---
RRF_K = 60
CANDIDATES = 20       # wide & cheap: BM25 + dense -> top-20
CONTEXT_K = 5         # read deep & few: chunks given to the generator
BM25_K1, BM25_B = 1.5, 0.75

# --- Agent ---
MAX_ITERATIONS = 8
MAX_TOKENS_PER_RUN = 60_000      # hard budget: the run stops gracefully above this
MAX_REPEATED_CALLS = 2           # identical (tool, args) calls tolerated before a nudge


def has_api_key() -> bool:
    return bool(os.environ.get(API_KEY_ENV))
