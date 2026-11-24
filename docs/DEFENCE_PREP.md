# Soutenance — 10 min de pitch + 5 min de Q&R (27 novembre)

## Déroulé du pitch (10 min)

| Temps | Qui | Contenu | Support |
|---|---|---|---|
| 0:00–2:00 | A | Le problème : une base interne de 500 décisions d'architecture (les PEPs). Pourquoi c'est dur : identifiants exacts, paraphrases, PEPs rejetées/remplacées quasi-identiques, questions en français. Protocole : dev (27) pour régler, test (52) + agent (24) held-out pour rapporter. | Slide corpus + slide protocole |
| 2:00–5:00 | B | Chunking (3 stratégies, n = ⌈L/(c−o)⌉), FAISS IndexFlatIP, BM25, RRF k = 60, rerank LLM. **Tableau P@5 / MRR** sur test ; l'en-tête contextuel est le gain principal ; BM25 = 0 en français. | Diagramme (partie ①) + tableau README §4.1 |
| 5:00–7:30 | D | L'agent : 5 outils, validation de schéma, boucle bornée, `final_answer` + sources vérifiées. Les 9 modes de panne → chacun testé. **Démo live** (2 questions, voir ci-dessous). | Diagramme (partie ②) + terminal |
| 7:30–10:00 | C | Instrumentation : tokens, latence, coût par appel / par question, traces JSONL. **Scorecard agent vs RAG** (succès, coût/question, p95, faithfulness). Limites et suite. | `results/agent_test.md` |

## Démo live (préparer un plan B : capture vidéo + traces déjà générées)

```bash
python -m pepcopilot ask "How many years separate the creation of PEP 3107 and PEP 484? Use the creation years."
#   -> get_pep(3107), get_pep(484), calculator(2014-2006), final_answer  => 8, sources [3107, 484]
python -m pepcopilot ask "What is the status of PEP 622, and which PEP replaced it?"
#   -> piège "Superseded" : la réponse doit citer PEP 634
```
Montrer ensuite la ligne `terminated_by=… iterations=… tokens=… cost=$… latency=…`.

## Questions probables du jury

1. *Pourquoi 200/50 et pas autre chose ?* — Baseline du cours ; on a comparé 3 stratégies et 3 tailles de section sur **dev** uniquement.
2. *Votre golden set est petit — vos écarts sont-ils significatifs ?* — Non pour < 0,05 ; on le dit (README §7). Les gros écarts (en-tête, français) le sont.
3. *Comment savez-vous que le test est vraiment held-out ?* — Fichiers séparés ; tous les réglages viennent de `results/retrieval_dev_*` ; l'historique Git le montre.
4. *Que se passe-t-il si l'API OpenAI tombe ?* — 5 essais avec backoff exponentiel, puis repli RAG ; si le repli échoue, message d'erreur explicite (test `test_llm_outage…`).
5. *Le modèle invente un outil ?* — Observation d'erreur listant les outils réels ; compté dans le taux d'appels hallucinés.
6. *Pourquoi un agent plutôt qu'un RAG simple ?* — Questions de comptage/métadonnées et multi-étapes (calcul) que la recherche seule ne résout pas ; comparer les deux lignes du scorecard (succès vs coût/latence).
7. *Pourquoi FAISS flat et pas HNSW ?* — ~9k vecteurs : la recherche exacte prend quelques ms ; HNSW/IVF ou pgvector au-delà de ~10⁶ ou pour la persistance et les filtres.
8. *Coût d'une question ?* — Lire `cost_usd_per_q` dans le scorecard (ordre de grandeur : quelques dixièmes de centime).
