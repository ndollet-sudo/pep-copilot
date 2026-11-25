"""Failure-injection suite: every failure mode of the agent, replayed deterministically."""
from conftest import FakeLLM, response, tool_call

from pepcopilot.agent.loop import Agent
from pepcopilot.llm import LLMError


def fallback(q):
    return {"answer": "RAG fallback answer [1]", "citations": [{"n": 1, "pep": 572}]}


def test_happy_path_search_then_final_answer(registry):
    llm = FakeLLM([response([tool_call("search_peps", {"query": "walrus"})]),
                   response([tool_call("final_answer", {"answer": ":= (PEP 572)", "sources": [572]})])])
    r = Agent(llm, registry).run("walrus?")
    assert r.terminated_by == "final_answer" and r.answer == ":= (PEP 572)"
    assert r.iterations == 2 and r.tool_calls == 2 and r.sources == [572]
    assert r.tokens_in == 200 and r.cost_usd > 0


def test_hallucinated_tool_then_recovery(registry):
    llm = FakeLLM([response([tool_call("serach_peps", {"query": "walrus"})]),
                   response([tool_call("search_peps", {"query": "walrus"})]),
                   response([tool_call("final_answer", {"answer": "ok", "sources": [572]})])])
    r = Agent(llm, registry).run("q")
    assert r.failures["unknown_tool"] == 1 and r.terminated_by == "final_answer"
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert "Available tools" in tool_msgs[0]["content"]          # the model was told what exists


def test_invalid_arguments_are_observations(registry):
    llm = FakeLLM([response([tool_call("get_pep", {"number": "abc"})]),
                   response([tool_call("final_answer", {"answer": "done"})])])
    r = Agent(llm, registry).run("q")
    assert r.failures["invalid_args"] == 1 and r.answer == "done"


def test_repeated_call_is_not_reexecuted(registry):
    same = {"expression": "1+1"}
    llm = FakeLLM([response([tool_call("calculator", same)])] * 3 +
                  [response([tool_call("final_answer", {"answer": "2"})])])
    r = Agent(llm, registry).run("q")
    assert r.failures["repeated_call"] == 1 and r.answer == "2"


def test_ungrounded_final_answer_is_rejected_once(registry):
    llm = FakeLLM([response([tool_call("final_answer", {"answer": "x", "sources": [484]})]),
                   response([tool_call("get_pep", {"number": 484})]),
                   response([tool_call("final_answer", {"answer": "x", "sources": [484]})])])
    r = Agent(llm, registry).run("q")
    assert r.failures["ungrounded_answer"] == 1
    assert r.terminated_by == "final_answer" and r.sources == [484]


def test_max_iterations_falls_back_to_rag(registry):
    llm = FakeLLM([response([tool_call("calculator", {"expression": f"{i}+1"})]) for i in range(10)])
    r = Agent(llm, registry, fallback=fallback, max_iterations=3).run("q")
    assert r.terminated_by == "fallback_rag" and r.iterations == 3 and r.sources == [572]


def test_llm_outage_falls_back_then_errors_cleanly(registry):
    r = Agent(FakeLLM([LLMError("503")]), registry, fallback=fallback).run("q")
    assert r.terminated_by == "fallback_rag" and r.failures["llm_error"] == 1
    r = Agent(FakeLLM([LLMError("503")]), registry, fallback=None).run("q")
    assert r.terminated_by == "error" and "could not complete" in r.answer


def test_token_budget(registry):
    big = [response([tool_call("calculator", {"expression": f"{i}*2"})], tin=50_000) for i in range(5)]
    r = Agent(FakeLLM(big), registry, fallback=fallback, max_tokens=60_000).run("q")
    assert r.failures["budget"] == 1 and r.terminated_by == "fallback_rag"


def test_soft_termination_plain_text(registry):
    r = Agent(FakeLLM([response(content="Plain answer.")]), registry).run("q")
    assert r.terminated_by == "text" and r.answer == "Plain answer."


def test_langgraph_port_same_behaviour(registry):
    from pepcopilot.agent.graph import run_graph
    llm = FakeLLM([response([tool_call("search_peps", {"query": "walrus"})]),
                   response([tool_call("final_answer", {"answer": ":=", "sources": [572]})])])
    out = run_graph(llm, registry, "walrus?")
    assert out["answer"] == ":=" and out["iterations"] == 2
