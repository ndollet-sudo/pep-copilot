import pytest

from pepcopilot.embeddings import LocalLSAEmbedder
from pepcopilot.eval.metrics import (doc_ranking, mrr, ndcg_at_k, precision_at_k, recall_at_k,
                                     relevant_ids)
from pepcopilot.retrieval import Retriever, rrf_fuse, tokenize


def test_rrf_deck_case():
    fused = rrf_fuse(["A", "B", "C"], ["B", "C", "d", "e", "f", "g", "h", "i", "A"])
    assert fused[:3] == ["B", "C", "A"]          # consensus beats one strong vote


def test_tokenize_keeps_dunders():
    assert "__exit__" in tokenize("How does __exit__ work?")
    assert "the" not in tokenize("the with statement")


def test_metrics_deck_mini_case():
    ranked, gold = ["x", "g1", "g2", "x", "x"], {"g1", "g2"}
    assert recall_at_k(ranked, gold, 5) == 1.0
    assert precision_at_k(ranked, gold, 5) == 0.40
    assert mrr(ranked, gold) == 0.5
    assert ndcg_at_k(ranked, gold, 5) == pytest.approx(0.69, abs=0.005)


def test_relevant_ids_and_doc_ranking():
    chunks = [{"chunk_id": "pep-0001#c0", "doc_id": "pep-0001", "text": "the walrus"},
              {"chunk_id": "pep-0001#c1", "doc_id": "pep-0001", "text": "nothing here"},
              {"chunk_id": "pep-0002#c0", "doc_id": "pep-0002", "text": "walrus too"}]
    assert relevant_ids(chunks, ["pep-0001"], "walrus") == {"pep-0001#c0"}
    assert relevant_ids(chunks, ["pep-0001"], "zzz") == {"pep-0001#c0", "pep-0001#c1"}
    assert doc_ranking(["pep-0002#c0", "pep-0001#c1", "pep-0002#c3"]) == ["pep-0002", "pep-0001"]


def test_hybrid_retriever_end_to_end_offline():
    texts = ["the walrus operator assigns inside expressions", "pattern matching with match case",
             "the global interpreter lock can be disabled", "type hints and the typing module"] * 3
    chunks = [{"chunk_id": f"d{i}#c0", "doc_id": f"d{i}", "pep": i, "title": "t", "status": "Final",
               "section": "", "text": t, "embed_text": t} for i, t in enumerate(texts)]
    r = Retriever(chunks, LocalLSAEmbedder(dim=4))
    top = r.search("disable the interpreter lock", mode="hybrid", k=3)
    assert "lock" in r.by_id[top[0]]["text"]
    assert len(r.search("walrus", mode="bm25", k=2)) == 2
