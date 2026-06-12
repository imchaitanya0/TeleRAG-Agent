"""
src/agent/nodes/reflect.py

REFLECT node: self-critiques the generated answer and decides what to do next.

Decision logic (from plan.md):
  confidence >= 0.8  →  output (DONE)
  0.5 <= conf < 0.8  →  re-plan with gap analysis (needs_retrieval)
  conf < 0.5         →  ask clarifying question (needs_clarification)

Confidence is computed without an LLM call using a fast heuristic
(token overlap between answer and context + source count).
This keeps the REFLECT step under 200ms.
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.agent.state import AgentState
from src.config import CONFIDENCE_THRESHOLD, CLARIFY_THRESHOLD, MAX_ITERATIONS


def _compute_confidence(answer: str, context: str, sources: list) -> float:
    """
    Fast heuristic confidence score (no LLM call):

    Factors:
    1. Source count  — more high-scoring sources → more confident
    2. Answer length — very short = uncertain
    3. Context overlap — fraction of 4+ char answer words found in context
    4. Hedging penalty — "I don't know", "unclear", etc.
    """
    if not answer or len(answer.strip()) < 20:
        return 0.0

    # Factor 1: source quality
    high_quality_sources = sum(1 for s in sources if s.get("score", 0) >= 0.5)
    source_score = min(high_quality_sources / max(len(sources), 1), 1.0)

    # Factor 2: answer length (normalised, capped at 1.0)
    length_score = min(len(answer.split()) / 50, 1.0)

    # Factor 3: context overlap
    stop_words = {"the", "a", "an", "is", "of", "in", "to", "and", "for",
                  "that", "this", "with", "by", "on", "are", "be", "it"}
    answer_words = {w.lower() for w in answer.split() if len(w) > 3 and w.lower() not in stop_words}
    ctx_lower = context.lower() if context else ""
    if answer_words:
        overlap = sum(1 for w in answer_words if w in ctx_lower)
        overlap_score = overlap / len(answer_words)
    else:
        overlap_score = 0.3  # neutral

    # Factor 4: hedging penalty
    hedges = ["i don't know", "i'm not sure", "unclear", "cannot determine",
              "no information", "not found", "unable to"]
    hedging_penalty = 0.3 if any(h in answer.lower() for h in hedges) else 0.0

    raw = (0.35 * source_score + 0.20 * length_score + 0.45 * overlap_score) - hedging_penalty
    return max(0.0, min(round(raw, 3), 1.0))


def _gap_analysis(query: str, answer: str, context: str) -> str:
    """
    Identify what the current answer is missing without an LLM call.
    Returns a short string describing the gap.
    """
    gaps = []

    # Check if query mentions specific spec numbers that aren't in the answer
    spec_refs = re.findall(r"(TS|TR)\s*(\d+\.\d+)", query, re.IGNORECASE)
    for ref_type, ref_num in spec_refs:
        spec = f"{ref_type} {ref_num}"
        if spec.lower() not in answer.lower() and spec.lower() not in context.lower():
            gaps.append(f"No information found for {spec}")

    # Check for clause references in query
    clause_refs = re.findall(r"§\s*([\d.]+)", query)
    for clause in clause_refs:
        if clause not in answer and clause not in context:
            gaps.append(f"Clause §{clause} not retrieved")

    if not gaps:
        gaps.append("The retrieved context may not fully cover the query — try broader sub-queries")

    return "; ".join(gaps)


def reflect_node(state: AgentState) -> AgentState:
    """
    REFLECT node: evaluate confidence and decide next action.

    Returns state with:
    - confidence: 0.0–1.0
    - reflection_notes: gap description
    - needs_clarification: True if confidence < CLARIFY_THRESHOLD
    - iteration: incremented
    """
    answer = state.get("answer", "")
    context = state.get("retrieved_context", "")
    sources = state.get("sources", [])
    query = state["query"]
    iteration = state.get("iteration", 0) + 1

    confidence = _compute_confidence(answer, context, sources)
    reflection_notes = _gap_analysis(query, answer, context)

    needs_clarification = (confidence < CLARIFY_THRESHOLD)

    # Build clarifying question if needed
    if needs_clarification:
        final_answer = (
            "I need a bit more context to give you an accurate answer.\n\n"
            f"Could you clarify:\n{reflection_notes}\n\n"
            "For example, are you asking about a specific 3GPP release, "
            "a particular network configuration, or a specific alarm type?"
        )
    else:
        final_answer = state.get("final_answer", answer)

    return {
        **state,
        "confidence": confidence,
        "reflection_notes": reflection_notes,
        "needs_clarification": needs_clarification,
        "iteration": iteration,
        "final_answer": final_answer,
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function for LangGraph.

    Returns:
    - "done"            → confidence >= 0.8 or max iterations reached
    - "retrieve"        → confidence 0.5–0.8 → re-retrieve with gap info
    - "clarify"         → confidence < 0.5 → ask user
    """
    confidence = state.get("confidence", 0.0)
    iteration = state.get("iteration", 0)
    needs_clarification = state.get("needs_clarification", False)

    if needs_clarification or confidence < CLARIFY_THRESHOLD:
        return "clarify"

    if confidence >= CONFIDENCE_THRESHOLD or iteration >= MAX_ITERATIONS:
        return "done"

    # Mid-confidence: try one more retrieval with gap analysis
    return "retrieve"
