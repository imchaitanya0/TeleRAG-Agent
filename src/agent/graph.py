"""
src/agent/graph.py

LangGraph StateGraph: wires the 4 nodes into the agentic loop.

    PLAN → RETRIEVE → GENERATE → REFLECT
                ↑                    |
                |      (mid-conf)    |
                └────────────────────┘
                         ↓ (done / clarify)
                       END
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes.plan import plan_node
from src.agent.nodes.retrieve import retrieve_node
from src.agent.nodes.generate import generate_node
from src.agent.nodes.reflect import reflect_node, should_continue


def build_agent_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Node execution order:
      1. plan     → classifies query, decomposes into sub-queries
      2. retrieve → hybrid search + rerank + context assembly
      3. generate → LLM answer generation with citations
      4. reflect  → confidence scoring + gap analysis
         → if done/clarify  → END
         → if needs retrieval → back to retrieve (max MAX_ITERATIONS times)
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────
    graph.add_node("plan",     plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reflect",  reflect_node)

    # ── Entry point ───────────────────────────────────────────
    graph.set_entry_point("plan")

    # ── Linear edges ──────────────────────────────────────────
    graph.add_edge("plan",     "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "reflect")

    # ── Conditional edge from REFLECT ─────────────────────────
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "done":     END,        # High confidence or max iterations
            "clarify":  END,        # Low confidence → ask user
            "retrieve": "retrieve", # Mid confidence → re-retrieve
        },
    )

    return graph.compile()


# ── Convenience: module-level compiled agent ───────────────────
_agent = None


def get_agent():
    """Return the singleton compiled agent graph."""
    global _agent
    if _agent is None:
        _agent = build_agent_graph()
    return _agent


def run_agent(query: str) -> dict:
    """
    Run the full agentic loop for a user query.

    Returns:
        {
            "final_answer":  str,   # Answer shown to user
            "answer":        str,   # Raw model answer (without citation block)
            "confidence":    float, # 0.0–1.0
            "sources":       list,  # [{spec, clause, title, score}, ...]
            "query_type":    str,
            "iteration":     int,
            "needs_clarification": bool,
        }
    """
    agent = get_agent()

    initial_state: AgentState = {
        "query":              query,
        "query_type":         "",
        "sub_queries":        [],
        "tools_to_use":       [],
        "retrieved_context":  "",
        "sources":            [],
        "retrieval_attempts": 0,
        "answer":             "",
        "citations":          [],
        "confidence":         0.0,
        "reflection_notes":   "",
        "iteration":          0,
        "needs_clarification": False,
        "final_answer":       "",
    }

    final_state = agent.invoke(initial_state)
    return {
        "final_answer":        final_state.get("final_answer", ""),
        "answer":              final_state.get("answer", ""),
        "confidence":          final_state.get("confidence", 0.0),
        "sources":             final_state.get("sources", []),
        "query_type":          final_state.get("query_type", ""),
        "iteration":           final_state.get("iteration", 0),
        "needs_clarification": final_state.get("needs_clarification", False),
        "reflection_notes":    final_state.get("reflection_notes", ""),
        "retrieved_context":   final_state.get("retrieved_context", ""),
    }


# ── Smoke test: python src/agent/graph.py ─────────────────────
if __name__ == "__main__":
    import time

    test_queries = [
        "What is RRC Connection Reconfiguration in 5G NR?",
        "Why is my cell experiencing high handover failure rate?",
        "How do I optimize DRX parameters for energy saving?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        t0 = time.perf_counter()
        result = run_agent(q)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"TYPE:       {result['query_type']}")
        print(f"CONFIDENCE: {result['confidence']:.2f}")
        print(f"ITERATIONS: {result['iteration']}")
        print(f"LATENCY:    {elapsed:.0f}ms")
        print(f"ANSWER:\n{result['final_answer'][:400]}")
        print("\nSOURCES:")
        for s in result["sources"][:3]:
            print(f"  {s['spec']} §{s['clause']}  [{s['score']}]")
