"""
src/ui/components.py

Reusable Gradio component builders for TeleRAG-Agent UI.

Provides:
  - format_sources_table()  → Gradio Dataframe for source citations
  - format_thinking_trace() → Accordion markdown for agent reasoning
  - confidence_label()      → Color-coded confidence text
  - format_alarm_result()   → Alarm analysis result display
  - format_kpi_result()     → KPI analysis result display
"""

import math


# ── Color / label helpers ──────────────────────────────────────

def confidence_label(confidence: float) -> str:
    """Return an emoji + color label for a confidence score."""
    if confidence >= 0.8:
        return f"🟢 High confidence ({confidence*100:.0f}%)"
    elif confidence >= 0.5:
        return f"🟡 Medium confidence ({confidence*100:.0f}%)"
    else:
        return f"🔴 Low confidence ({confidence*100:.0f}%) — answer may be incomplete"


def query_type_badge(query_type: str) -> str:
    """Return an emoji label for the detected query type."""
    badges = {
        "spec_qa":      "📋 Specification Q&A",
        "troubleshoot": "🔧 Troubleshooting",
        "optimization": "⚡ Optimization",
        "kpi":          "📊 KPI Analysis",
        "general":      "💬 General",
    }
    return badges.get(query_type, f"❓ {query_type}")


# ── Source table ───────────────────────────────────────────────

def format_sources_table(sources: list[dict]) -> list[list]:
    """
    Format sources list into a 2D list for a Gradio Dataframe.

    Input:  [{spec, clause, title, score}, ...]
    Output: [[#, Specification, Clause, Title, Relevance], ...]
    """
    rows = []
    for i, s in enumerate(sources, 1):
        spec = s.get("spec", "")
        clause = s.get("clause", "")
        title = s.get("title", "")[:60] + ("..." if len(s.get("title", "")) > 60 else "")
        score = s.get("score", 0.0)
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        rows.append([i, spec, f"§{clause}" if clause else "—", title, f"{bar} {score:.3f}"])
    return rows


SOURCES_HEADERS = ["#", "Specification", "Clause", "Title", "Relevance"]


# ── Thinking trace ─────────────────────────────────────────────

def format_thinking_trace(agent_result: dict) -> str:
    """
    Build a markdown string showing the agent's step-by-step reasoning.
    Displayed inside a Gradio Accordion.
    """
    lines = []

    # PLAN
    lines.append("### 🗺️ Step 1: PLAN")
    lines.append(f"**Query type detected:** {query_type_badge(agent_result.get('query_type', ''))}")
    sub_queries = agent_result.get("sub_queries_used", [])
    if sub_queries:
        lines.append(f"**Sub-queries generated:**")
        for i, sq in enumerate(sub_queries, 1):
            lines.append(f"  {i}. _{sq}_")

    # RETRIEVE
    lines.append("\n### 🔍 Step 2: RETRIEVE")
    sources = agent_result.get("sources", [])
    lines.append(f"**Passages retrieved:** {len(sources)}")
    if sources:
        top = sources[0]
        lines.append(f"**Top source:** {top.get('spec','')} §{top.get('clause','')} "
                     f"— _{top.get('title','')[:50]}_ (score: {top.get('score', 0):.3f})")

    # GENERATE
    lines.append("\n### 🤖 Step 3: GENERATE")
    answer_preview = agent_result.get("answer", "")[:200]
    if answer_preview:
        lines.append(f"**Answer preview:** _{answer_preview}..._")

    # REFLECT
    lines.append("\n### 🪞 Step 4: REFLECT")
    confidence = agent_result.get("confidence", 0.0)
    iterations = agent_result.get("iteration", 1)
    lines.append(f"**Confidence score:** {confidence_label(confidence)}")
    lines.append(f"**Agent iterations:** {iterations}")
    if agent_result.get("reflection_notes"):
        lines.append(f"**Gap analysis:** _{agent_result['reflection_notes']}_")

    if agent_result.get("needs_clarification"):
        lines.append("\n⚠️ **Agent needs clarification** — asked a follow-up question.")

    return "\n".join(lines)


# ── Alarm result formatter ─────────────────────────────────────

def format_alarm_result(result: dict) -> str:
    """Format alarm_analyzer_tool output as markdown."""
    if not result or result.get("filtered_alarms", 0) == 0:
        return "_No alarms found for the specified filters._"

    lines = [
        f"### 🚨 O-RAN Alarm Analysis",
        f"**Total alarms in system:** {result.get('total_alarms', 0):,}",
        f"**Alarms matching filter:** {result.get('filtered_alarms', 0):,}",
        "",
        "**Severity distribution:**",
    ]
    for sev, count in sorted(result.get("severity_dist", {}).items()):
        emoji = {"critical": "🔴", "major": "🟠", "minor": "🟡", "warning": "⚪"}.get(sev, "⚫")
        lines.append(f"  - {emoji} {sev.capitalize()}: {count}")

    storms = result.get("storms_detected", [])
    if storms:
        lines.append(f"\n**⚡ Alarm storms detected: {len(storms)}**")
        for storm in storms[:3]:
            lines.append(
                f"  - Cell **{storm['cell_id']}**: {storm['alarm_count']} alarms "
                f"({storm['dominant_type']}) — causes: {', '.join(storm.get('probable_causes', [])[:2])}"
            )

    lines.append(f"\n**Root Cause Analysis:**\n```\n{result.get('rca_summary', '')}\n```")
    return "\n".join(lines)


# ── KPI result formatter ───────────────────────────────────────

def format_kpi_result(result: dict) -> str:
    """Format kpi_calculator_tool output as markdown."""
    if not result:
        return "_No KPI data available._"

    status_emoji = {"CRITICAL": "🔴", "DEGRADED": "🟡", "NORMAL": "🟢"}
    lines = [
        "### 📊 KPI Analysis",
        f"**Cell:** {result.get('cell_id', 'all')}",
        "",
        "**KPI Status:**",
    ]
    for kpi, status in result.get("kpi_status", {}).items():
        emoji = status_emoji.get(status, "⚫")
        stats = result.get("summary_stats", {}).get(kpi, {})
        lines.append(
            f"  - {emoji} **{kpi}**: {status} "
            f"(mean={stats.get('mean','?')}, {stats.get('pct_in_good_range','?')}% in good range)"
        )

    anomalies = result.get("anomalies", [])
    if anomalies:
        lines.append(f"\n**Top {len(anomalies)} Anomalies:**")
        for a in anomalies:
            lines.append(
                f"  - {a['kpi']} on {a['cell_id']}: "
                f"value={a['value']}{a['unit']}, z={a['z_score']:.2f} "
                f"(spec: {a['threshold_info']['spec']})"
            )

    lines.append(f"\n**Analysis:**\n```\n{result.get('analysis_report', '')}\n```")
    return "\n".join(lines)
