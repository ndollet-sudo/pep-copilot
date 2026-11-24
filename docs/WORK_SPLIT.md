# Répartition du travail — 4 membres

Remplacez « Membre A/B/C/D » par vos prénoms (ici et dans le README).
Chaque membre **committe lui-même, depuis sa machine, avec son identité Git** — l'historique
doit refléter qui a porté quoi (le jury et la note individuelle s'appuient dessus).

| Membre | Rôle | Fichiers dont il/elle est propriétaire | Partie du pitch |
|---|---|---|---|
| **A** | Données & évaluation | `scripts/build_corpus.py`, `corpus.py`, `config.py`, golden sets dev/test, `eval/metrics.py`, `eval/retrieval_eval.py`, README | Problème, corpus, protocole held-out (≈2 min) |
| **B** | Retrieval | `chunking.py`, `embeddings.py`, `retrieval.py` (BM25, FAISS, RRF), `rerank.py`, scorecards retrieval | Chunking + FAISS/BM25/RRF + tableau P@k / MRR (≈3 min) |
| **C** | LLM, génération, instrumentation | `llm.py` (backoff), `telemetry.py`, `generate.py`, `pipeline.py`, CLI, `eval/agent_eval.py`, `eval/faithfulness.py` | Instrumentation tokens/latence/coût + scorecard agent vs RAG (≈2,5 min) |
| **D** | Agent | `agent/tools.py`, `agent/loop.py`, `agent/graph.py`, tests d'injection de pannes, diagramme d'architecture | Workflow agentique, sélection d'outils, gestion des pannes + démo live (≈2,5 min) |

## Commits — un par personne

Chaque membre reçoit **son kit** (`membre_A.zip`, …) : ses fichiers uniquement + `team_commit.py`,
qui fait **un** commit sous **sa** propre identité Git (pull → copie → add → commit → push).

1. **A** crée un dépôt **vide** (GitHub/GitLab, sans README), invite B, C, D **et l'enseignant**.
2. Chacun : `git config --global user.name "Prénom Nom"`, `git config --global user.email "..."`,
   puis `git clone URL_DU_DEPOT`.
3. **A d'abord**, puis B, C, D dans n'importe quel ordre :
   `python team_commit.py --repo CHEMIN/VERS/SON/CLONE`
4. Ensuite : `git pull` puis `make test` → **27 passed** chez tout le monde.

| Membre | Commit |
|---|---|
| A | corpus 500 PEPs, golden sets, métriques, README, configuration |
| B | chunking, BM25 + FAISS + RRF, reranker, scorecards hors-ligne |
| C | client OpenAI + backoff, télémétrie, génération citée, pipeline RAG, éval bout-en-bout |
| D | outils, boucle agent + gestion des pannes, LangGraph, diagramme d'architecture |

`python scripts/commit_plan.py --check` vérifie que chaque fichier appartient à un seul commit.

## Environnement

```bash
make setup            # crée .venv, installe les dépendances, crée .env à partir du modèle
```
Sous Windows : `python -m venv .venv`, `.venv\Scripts\Activate.ps1`, `pip install -r requirements.txt`,
`copy .env.example .env`. Chacun crée son propre `.venv` et son propre `.env` : ni l'un ni l'autre
n'est committé (`.gitignore`). **Ne jamais committer la clé.**

## Runs payants (une seule clé suffit, coût total < 0,5 $) — après les 4 commits

```bash
# dans .env : OPENAI_API_KEY=sk-...
make eval-retrieval   # B — ~0,2 $ (embedding du corpus ×3 stratégies + rerank), ~15 min
make eval-agent       # C — ~0,2 $, ~10 min
```
Puis la personne qui a lancé les runs committe les résultats :
`git add results && git commit -m "results: API scorecards" && git push`.
Si les chiffres avec l'embedder OpenAI changent le meilleur chunking sur **dev**, changez `DEFAULT_STRATEGY`
dans `config.py` (choix sur dev, report sur test — jamais l'inverse).

## Ce que chacun doit savoir expliquer seul (note individuelle + Q&R)

- **A** : pourquoi les PEPs Rejected/Superseded restent dans l'index ; règle de pertinence
  (PEP gold **et** motif de preuve) ; pourquoi dev ≠ test ; différence Recall@k / precision@k / MRR / nDCG.
- **B** : formule n = ⌈L/(c−o)⌉ et coût de l'overlap ; pourquoi l'en-tête contextuel aide ;
  RRF avec k = 60 (exemple A/B/C du cours) ; IndexFlatIP = cosinus sur vecteurs normalisés ;
  quand passer à IVF/HNSW ou pgvector ; pourquoi BM25 = 0 sur les questions en français.
- **C** : politique de retry (2, 4, 8, 16 s ; 4xx = échec immédiat) ; comment le coût est calculé
  (usage × table de prix) ; p50/p95 ; faithfulness RAGAS (décomposer → vérifier → ratio) ;
  JSON mode + boucle de réparation Pydantic.
- **D** : boucle ReAct, `final_answer` + borne d'itérations ; les 9 modes de panne et leur test ;
  contrôle d'ancrage des sources ; repli vers le RAG ; équivalence boucle manuelle ↔ LangGraph.
