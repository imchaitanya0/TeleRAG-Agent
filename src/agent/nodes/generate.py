"""
src/agent/nodes/generate.py

GENERATE node: builds a context-grounded prompt and calls the LLM
to produce a faithful, cited answer.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.agent.state import AgentState
from src.agent.prompts import GENERATION_PROMPT, GENERATION_PROMPT_NO_CONTEXT
from src.models.inference import generate


def generate_node(state: AgentState) -> AgentState:
    """
    GENERATE node: produces a grounded, cited answer.

    - Uses context from RETRIEVE if available.
    - Falls back to knowledge-only generation if context is empty.
    - Appends citations from sources list to the final answer.
    """
    query = state["query"]
    context = state.get("retrieved_context", "")
    citations = state.get("citations", [])

    # ── Build prompt ───────────────────────────────────────────
    if context and len(context.strip()) > 100:
        prompt = GENERATION_PROMPT.format(
            context=context[:5000],   # Stay within model context window
            query=query,
        )
    else:
        prompt = GENERATION_PROMPT_NO_CONTEXT.format(query=query)

    # ── Generate ───────────────────────────────────────────────
    raw_answer = generate(
        prompt,
        max_new_tokens=400,
        do_sample=False,
        repetition_penalty=1.1,
    )

    # ── Append source citations ────────────────────────────────
    if citations:
        citation_block = "\n\n**Sources:** " + " | ".join(citations[:5])
        final_answer_text = raw_answer.strip() + citation_block
    else:
        final_answer_text = raw_answer.strip()

    return {
        **state,
        "answer": raw_answer.strip(),
        "final_answer": final_answer_text,
    }
