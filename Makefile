export PYTHONPATH := src
VENV := .venv
# use the project's virtual environment when it exists
PY ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

.PHONY: setup install corpus test eval-offline eval-retrieval eval-agent ask

setup:                  ## create .venv, install dependencies, create .env from the template
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "-> .env created: put your OPENAI_API_KEY in it")

install:
	$(PY) -m pip install -r requirements.txt

corpus:                 ## rebuild data/corpus/peps.jsonl from github.com/python/peps
	$(PY) scripts/build_corpus.py

test:                   ## unit + failure-injection tests (offline, no API key)
	$(PY) -m pytest -q tests

eval-offline:           ## retrieval scorecard with the local LSA embedder (free)
	$(PY) -m pepcopilot.eval.retrieval_eval --split dev  --embedder lsa
	$(PY) -m pepcopilot.eval.retrieval_eval --split test --embedder lsa

eval-retrieval:         ## retrieval scorecard with the API embedder + LLM rerank (needs key, ~$0.2)
	$(PY) -m pepcopilot.eval.retrieval_eval --split dev  --embedder api --rerank
	$(PY) -m pepcopilot.eval.retrieval_eval --split test --embedder api --rerank

eval-agent:             ## agent vs RAG-only on the held-out agent set (needs key, ~$0.2)
	$(PY) -m pepcopilot.eval.agent_eval --compare-rag

ask:                    ## make ask Q="Which PEP introduced except*?"
	$(PY) -m pepcopilot ask "$(Q)"
