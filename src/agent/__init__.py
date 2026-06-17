"""
src/agent/__init__.py

Lazy imports to avoid triggering the full model/embedding load chain
at import time (which would crash Kaggle before the UI even starts).
Models are only loaded when first_called.
"""

# Do NOT do eager imports here — they trigger sentence_transformers,
# bitsandbytes, etc. at import time which can crash in environments
# where those libraries are not yet properly installed.
# Instead, expose lazy wrappers.

def run_agent(query: str) -> dict:
    from src.agent.graph import run_agent as _run_agent
    return _run_agent(query)


def get_agent():
    from src.agent.graph import get_agent as _get_agent
    return _get_agent()


def build_agent_graph():
    from src.agent.graph import build_agent_graph as _build_agent_graph
    return _build_agent_graph()


# AgentState is a TypedDict — safe to import eagerly (no side effects)
from src.agent.state import AgentState  # noqa: E402

__all__ = ["run_agent", "get_agent", "build_agent_graph", "AgentState"]
