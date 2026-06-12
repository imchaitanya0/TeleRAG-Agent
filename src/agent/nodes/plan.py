"""
src/agent/nodes/plan.py

PLAN node: classifies the query type and decomposes complex queries
into focused sub-queries for the RETRIEVE node.

Uses keyword-based classification first (fast, no LLM call).
Falls back to LLM classification for ambiguous queries.
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.agent.state import AgentState
from src.agent.prompts import CLASSIFICATION_PROMPT
from src.models.inference import generate

# ──────────────────────────────────────────────────────────────
# Telecom glossary for keyword-based fast classification
# ──────────────────────────────────────────────────────────────

_SPEC_KEYWORDS = {
    "ts ", "tr ", "3gpp", "release", "clause", "§", "section",
    "38.331", "38.300", "38.321", "38.322", "23.501", "23.502", "38.213",
    "rrc", "pdcp", "rlc", "mac", "harq", "prach", "pucch", "pusch", "pdsch",
    "nr ", "lte", "5g", "4g", "ue ", "gnb", "enodeb",
}

_TROUBLESHOOT_KEYWORDS = {
    "failure", "failed", "error", "fault", "alarm", "issue", "problem",
    "root cause", "rca", "debug", "outage", "degraded", "drop", "lost",
    "handover failure", "ho failure", "connection loss", "unavailable",
}

_OPTIMIZATION_KEYWORDS = {
    "optimize", "optimise", "tune", "improve", "enhance", "reduce latency",
    "increase throughput", "energy saving", "parameter", "configuration",
    "drx", "cqi", "mcs", "beamforming", "load balancing",
}

_KPI_KEYWORDS = {
    "kpi", "rsrp", "rsrq", "sinr", "throughput", "latency", "jitter",
    "packet loss", "availability", "reliability", "anomaly", "threshold",
    "measurement", "counter", "metric", "performance",
}


def _keyword_classify(query: str) -> str:
    q = query.lower()
    scores = {
        "spec_qa": sum(1 for kw in _SPEC_KEYWORDS if kw in q),
        "troubleshoot": sum(1 for kw in _TROUBLESHOOT_KEYWORDS if kw in q),
        "optimization": sum(1 for kw in _OPTIMIZATION_KEYWORDS if kw in q),
        "kpi": sum(1 for kw in _KPI_KEYWORDS if kw in q),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _simple_decompose(query: str, query_type: str) -> list[str]:
    """
    Rule-based sub-query generation — fast and deterministic.
    Supplements LLM decomposition for well-understood query types.
    """
    sub_queries = [query]  # Always include the original

    q_lower = query.lower()

    if query_type == "troubleshoot":
        sub_queries.append(f"3GPP specification for {query}")
        sub_queries.append(f"common causes of {query}")

    elif query_type == "optimization":
        sub_queries.append(f"3GPP parameters for {query}")
        sub_queries.append(f"best practices {query}")

    elif query_type == "kpi":
        sub_queries.append(f"KPI threshold definition {query}")
        sub_queries.append(f"measurement procedure {query}")

    elif query_type == "spec_qa":
        # Extract spec number if present
        spec_match = re.search(r"(TS|TR)\s*([\d.]+)", query, re.IGNORECASE)
        if spec_match:
            spec = f"{spec_match.group(1)} {spec_match.group(2)}"
            sub_queries.append(f"{spec} clause definition")

    return list(dict.fromkeys(sub_queries))  # deduplicate preserving order


def plan_node(state: AgentState) -> AgentState:
    """
    PLAN node: classify the query and decompose into sub-queries.

    Fast-path: keyword classification (no LLM call) + rule-based decomposition.
    This keeps the PLAN step under 100ms.
    """
    query = state["query"]

    # 1. Classify
    query_type = _keyword_classify(query)

    # 2. Decompose
    sub_queries = _simple_decompose(query, query_type)

    # 3. Select tools based on query type
    tools_map = {
        "spec_qa":      ["spec_retriever"],
        "troubleshoot": ["spec_retriever", "alarm_analyzer"],
        "optimization": ["spec_retriever", "kpi_calculator"],
        "kpi":          ["kpi_calculator", "spec_retriever"],
        "general":      ["spec_retriever"],
    }
    tools_to_use = tools_map.get(query_type, ["spec_retriever"])

    return {
        **state,
        "query_type": query_type,
        "sub_queries": sub_queries,
        "tools_to_use": tools_to_use,
        "iteration": state.get("iteration", 0),
        "retrieval_attempts": state.get("retrieval_attempts", 0),
        "needs_clarification": False,
    }
