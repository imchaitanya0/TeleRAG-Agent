"""
src/agent/state.py

Defines the shared state object that flows through every node of the
LangGraph PLAN → RETRIEVE → GENERATE → REFLECT agentic loop.
"""
from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Input ────────────────────────────────────────────
    query: str                          # Original user query

    # ── Planning ─────────────────────────────────────────
    query_type: str                     # "spec_qa" | "troubleshoot" | "optimization" | "kpi" | "general"
    sub_queries: list[str]              # Decomposed sub-queries for multi-hop retrieval
    tools_to_use: list[str]             # Tools selected by PLAN node

    # ── Retrieval ────────────────────────────────────────
    retrieved_context: str              # Assembled context string
    sources: list[dict]                 # [{spec, clause, title, score}, ...]
    retrieval_attempts: int             # How many retrieval rounds done

    # ── Generation ───────────────────────────────────────
    answer: str                         # Model-generated answer
    citations: list[str]                # ["TS 38.331 §5.3.3", ...]

    # ── Reflection ───────────────────────────────────────
    confidence: float                   # 0.0 – 1.0 self-assessed confidence
    reflection_notes: str               # Gap analysis from REFLECT node
    iteration: int                      # Current loop count (max MAX_ITERATIONS)
    needs_clarification: bool           # True → ask user for clarification

    # ── Final output ────────────────────────────────────
    final_answer: str                   # The answer shown to the user
