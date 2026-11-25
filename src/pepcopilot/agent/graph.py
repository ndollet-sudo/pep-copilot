"""The same agent as a LangGraph StateGraph (session 6, lab 2) - for comparison.

| Hand loop (loop.py)   | LangGraph                                         |
|-----------------------|---------------------------------------------------|
| messages list         | AgentState, persisted between nodes               |
| for-loop              | conditional edge agent -> tools -> agent          |
| max_iterations        | recursion_limit = 2 * max_iterations + 1          |
| registry.dispatch     | unchanged, called by tools_node                   |

The hand loop stays the reference implementation (it carries the grounding check and the
RAG fallback); the graph shows the scaffolding changes, the failure modes do not.
"""
from __future__ import annotations

from typing import TypedDict

from .. import config
from .loop import SYSTEM_PROMPT, _assistant_dict
from .tools import ToolRegistry, validate_args


class AgentState(TypedDict, total=False):
    messages: list
    n_agent_calls: int
    final_answer: str | None
    sources: list
    tool_calls: int
    hallucinated: int


def build_graph(llm, registry: ToolRegistry):
    from langgraph.graph import END, START, StateGraph

    def agent_node(state: AgentState) -> dict:
        resp = llm.chat(state["messages"], tools=registry.schemas, name="graph_agent_step")
        msg = resp.choices[0].message
        final, sources = None, state.get("sources", [])
        hallucinated = state["hallucinated"]
        calls = msg.tool_calls or []
        if not calls:
            final = msg.content
        for c in calls:
            if registry.classify(c.function.name, c.function.arguments) != "ok":
                hallucinated += 1
            elif c.function.name == "final_answer":
                _, args, _ = validate_args(registry.params("final_answer"), c.function.arguments)
                final, sources = args["answer"], args.get("sources", [])
        return {"messages": state["messages"] + [_assistant_dict(msg)],
                "n_agent_calls": state["n_agent_calls"] + 1, "final_answer": final,
                "sources": sources, "tool_calls": state["tool_calls"] + len(calls),
                "hallucinated": hallucinated}

    def tools_node(state: AgentState) -> dict:
        messages = list(state["messages"])
        for c in messages[-1].get("tool_calls", []):
            res = registry.dispatch(c["function"]["name"], c["function"]["arguments"])
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": res.observation})
        return {"messages": messages}

    def should_continue(state: AgentState):
        return END if state.get("final_answer") is not None else "tools"

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def run_graph(llm, registry: ToolRegistry, question: str,
              max_iterations: int = config.MAX_ITERATIONS) -> dict:
    from langgraph.errors import GraphRecursionError

    state = {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": question}],
             "n_agent_calls": 0, "final_answer": None, "sources": [], "tool_calls": 0,
             "hallucinated": 0}
    try:
        out = build_graph(llm, registry).invoke(
            state, config={"recursion_limit": 2 * max_iterations + 1})
    except GraphRecursionError:
        return {"answer": None, "terminated_by": "recursion_limit"}
    return {"answer": out["final_answer"], "sources": out["sources"],
            "terminated_by": "final_answer", "iterations": out["n_agent_calls"],
            "tool_calls": out["tool_calls"], "hallucinated_calls": out["hallucinated"]}
