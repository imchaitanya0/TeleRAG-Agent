"""
src/ui/app.py

TeleRAG-Agent: Gradio Chat Interface

Features:
  - Natural language chat interface
  - Real-time "thinking trace" accordion showing PLAN/RETRIEVE/GENERATE/REFLECT steps
  - Source citation panel with relevance scores
  - Confidence bar (color-coded: green/yellow/red)
  - Query type badge
  - O-RAN Alarm and KPI analysis tabs
  - Security: input sanitized before every query
  - Audit: every session logged to logs/audit.jsonl

Launch:
    source .venv/bin/activate
    python src/ui/app.py

Production (Kaggle/server):
    python src/ui/app.py --server_port 7860 --share
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import gradio as gr

from src.agent.graph import run_agent
from src.agent.tools.alarm_analyzer import alarm_analyzer_tool
from src.agent.tools.kpi_calculator import kpi_calculator_tool
from src.security.input_sanitizer import sanitize_input, InputSanitizationError
from src.security.pii_detector import detect_and_redact
from src.security.audit_logger import get_audit_logger
from src.ui.components import (
    format_sources_table,
    format_thinking_trace,
    confidence_label,
    query_type_badge,
    format_alarm_result,
    format_kpi_result,
    SOURCES_HEADERS,
)

# ── Audit logger ───────────────────────────────────────────────
_audit = get_audit_logger()

# ── Example queries ────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "What is RRC Connection Reconfiguration in 5G NR?",
    "Explain DRX operation and its parameters in LTE.",
    "What is the role of AMF in the 5G core network?",
    "Why is my cell experiencing high handover failure rate?",
    "How do I optimize RSRP thresholds to reduce unnecessary handovers?",
    "What is HARQ and how does it improve reliability in NR?",
    "What are the 3GPP specifications for O-RAN architecture?",
]

# ── Custom CSS ─────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global ── */
body, .gradio-container { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto; }

/* ── Header ── */
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 16px;
    border: 1px solid rgba(99,179,237,0.2);
}
.header-box h1 { color: #e2e8f0; font-size: 2rem; margin: 0; }
.header-box p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.95rem; }

/* ── Confidence bar ── */
.conf-high { color: #22c55e; font-weight: 600; font-size: 1.0rem; }
.conf-mid  { color: #eab308; font-weight: 600; font-size: 1.0rem; }
.conf-low  { color: #ef4444; font-weight: 600; font-size: 1.0rem; }

/* ── Chat messages ── */
.message-bot { background: #1e293b !important; border-radius: 8px !important; }
.message-user { background: #0f4c81 !important; border-radius: 8px !important; }

/* ── Source table ── */
.source-table { font-size: 0.85rem !important; }

/* ── Accordion ── */
.thinking-trace { background: #0f172a !important; border: 1px solid #334155 !important; }

/* ── Tabs ── */
.tab-nav button { font-weight: 600 !important; }

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
}
"""

# ── Core chat handler ──────────────────────────────────────────

def chat(
    user_message: str,
    history: list,
):
    """
    Main handler called on every user message.
    Yields (updated_history, thinking_trace, sources_table, confidence_label, query_type_label)
    """
    if not user_message or not user_message.strip():
        yield history, "_(empty query)_", [], "—", "—"
        return

    # ── 1. Security: sanitize input ───────────────────────────
    session_id = _audit.start_session(user_message)
    try:
        clean_query = sanitize_input(user_message)
    except InputSanitizationError as e:
        _audit.log_security_event(session_id, "query_rejected", {"reason": str(e)})
        _audit.log_completion(session_id, confidence=0.0, iterations=0)
        error_msg = f"⚠️ **Security check:** {e}"
        yield history + [[user_message, error_msg]], "_(query rejected by security filter)_", [], "0%", "🔒 Blocked"
        return

    # ── 2. Show "thinking" placeholder immediately ────────────
    thinking_msg = "_⏳ Thinking... (PLAN → RETRIEVE → GENERATE → REFLECT)_"
    yield (
        history + [[user_message, thinking_msg]],
        "_Running agent..._",
        [],
        "—",
        "—",
    )

    # ── 3. Run agent ──────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = run_agent(clean_query)
    except Exception as e:
        _audit.log_error(session_id, str(e))
        _audit.log_completion(session_id, confidence=0.0, iterations=0)
        error_msg = f"❌ **Agent error:** {str(e)[:200]}"
        yield history + [[user_message, error_msg]], f"_Error: {e}_", [], "0%", "❌"
        return

    latency_ms = (time.perf_counter() - t0) * 1000

    # ── 4. Security: PII redaction on output ─────────────────
    final_answer = result.get("final_answer", "No answer generated.")
    pii_result = detect_and_redact(final_answer)
    if pii_result.has_pii:
        _audit.log_security_event(session_id, "pii_detected", {
            "findings": [f["name"] for f in pii_result.findings]
        })
    clean_answer = pii_result.redacted_text

    # ── 5. Format answer ──────────────────────────────────────
    confidence = result.get("confidence", 0.0)
    iterations = result.get("iteration", 1)
    query_type = result.get("query_type", "")
    sources = result.get("sources", [])

    latency_line = f"\n\n---\n_⏱️ Response time: {latency_ms:.0f}ms | Agent iterations: {iterations}_"
    full_answer = clean_answer + latency_line

    # ── 6. Build UI components ────────────────────────────────
    thinking_md = format_thinking_trace({
        **result,
        "sub_queries_used": result.get("sub_queries", [clean_query]),
    })
    sources_table = format_sources_table(sources)
    conf_label = confidence_label(confidence)
    type_badge = query_type_badge(query_type)

    # ── 7. Audit completion ───────────────────────────────────
    _audit.log_completion(
        session_id,
        confidence=confidence,
        iterations=iterations,
        query_type=query_type,
        sources_used=len(sources),
        answer_length=len(clean_answer),
    )

    updated_history = history + [[user_message, full_answer]]
    yield updated_history, thinking_md, sources_table, conf_label, type_badge


# ── Alarm analysis handler ─────────────────────────────────────

def run_alarm_analysis(cell_id: str, severity: str, top_n: int):
    cell = cell_id.strip() if cell_id.strip() else None
    sev = severity if severity != "All" else None
    result = alarm_analyzer_tool(cell_id=cell, severity=sev, top_n=int(top_n))
    return format_alarm_result(result)


# ── KPI analysis handler ───────────────────────────────────────

def run_kpi_analysis(cell_id: str, kpi_name: str, top_n: int):
    cell = cell_id.strip() if cell_id.strip() else None
    kpi = kpi_name if kpi_name != "All KPIs" else None
    result = kpi_calculator_tool(cell_id=cell, kpi_name=kpi, top_n=int(top_n))
    return format_kpi_result(result)


# ── Build Gradio UI ────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="TeleRAG-Agent — 5G Telecom AI Assistant",
    ) as demo:

        # ── Header ────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <h1>📡 TeleRAG-Agent</h1>
            <p>AI assistant for 3GPP specifications, O-RAN fault analysis, and network KPI monitoring</p>
        </div>
        """)

        with gr.Tabs():

            # ─────────────── TAB 1: Chat ─────────────────────
            with gr.TabItem("💬 Ask a Question"):

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="TeleRAG-Agent",
                            height=500,
                            show_label=False,
                            elem_classes=["chat-window"],
                        )

                    with gr.Column(scale=1):
                        conf_label = gr.Markdown("**Confidence:** —", label="Confidence")
                        type_badge = gr.Markdown("**Query Type:** —", label="Query Type")

                        gr.Markdown("### 📚 Sources")
                        sources_table = gr.Dataframe(
                            headers=SOURCES_HEADERS,
                            datatype=["number", "str", "str", "str", "str"],
                            label="Retrieved Passages",
                            interactive=False,
                            wrap=True,
                            elem_classes=["source-table"],
                            max_height=300,
                        )

                # ── Input row ────────────────────────────────
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask about 5G NR, LTE specs, O-RAN faults, or network KPIs...",
                        show_label=False,
                        scale=5,
                        container=False,
                        max_lines=3,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80)

                # ── Thinking trace ────────────────────────────
                with gr.Accordion("🧠 Agent Thinking Trace", open=False, elem_classes=["thinking-trace"]):
                    thinking_trace = gr.Markdown("_Send a message to see the agent's reasoning steps._")

                # ── Example queries ───────────────────────────
                gr.Markdown("**Try an example:**")
                gr.Examples(
                    examples=EXAMPLE_QUERIES,
                    inputs=msg_box,
                    label="",
                )

                # ── Clear button ──────────────────────────────
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

                # ── Event wiring ──────────────────────────────
                submit_inputs  = [msg_box, chatbot]
                submit_outputs = [chatbot, thinking_trace, sources_table, conf_label, type_badge]

                send_btn.click(
                    fn=chat,
                    inputs=submit_inputs,
                    outputs=submit_outputs,
                ).then(lambda: "", None, msg_box)  # clear input

                msg_box.submit(
                    fn=chat,
                    inputs=submit_inputs,
                    outputs=submit_outputs,
                ).then(lambda: "", None, msg_box)

                clear_btn.click(
                    fn=lambda: ([], "_Cleared._", [], "—", "—"),
                    outputs=submit_outputs,
                )

            # ──────────── TAB 2: Alarm Analysis ─────────────
            with gr.TabItem("🚨 O-RAN Alarm Analysis"):
                gr.Markdown("""
                ### Real O-RAN Network Fault Analysis
                Analyses alarm events from real O-RAN testbed data (NetsLab-5GORAN-IDD) 
                and synthetic alarm logs. Detects alarm storms and identifies probable root causes.
                """)
                with gr.Row():
                    alarm_cell = gr.Textbox(
                        label="Cell ID (optional)",
                        placeholder="e.g. CELL_001, CELL_005",
                        scale=2,
                    )
                    alarm_severity = gr.Dropdown(
                        choices=["All", "critical", "major", "minor", "warning"],
                        value="All",
                        label="Severity Filter",
                        scale=1,
                    )
                    alarm_top_n = gr.Slider(
                        minimum=3, maximum=10, value=5, step=1,
                        label="Top N results",
                        scale=1,
                    )
                alarm_btn = gr.Button("🔍 Analyse Alarms", variant="primary")
                alarm_output = gr.Markdown("_Click 'Analyse Alarms' to run analysis._")
                alarm_btn.click(
                    fn=run_alarm_analysis,
                    inputs=[alarm_cell, alarm_severity, alarm_top_n],
                    outputs=alarm_output,
                )

            # ──────────── TAB 3: KPI Analysis ───────────────
            with gr.TabItem("📊 KPI Anomaly Detection"):
                gr.Markdown("""
                ### Network KPI Anomaly Detection
                Uses Z-score statistical analysis on cell-level KPI time-series.  
                Detects RSRP degradation, SINR drops, and RRC success rate anomalies  
                aligned with 3GPP TS 28.552 thresholds.
                """)
                with gr.Row():
                    kpi_cell = gr.Textbox(
                        label="Cell ID (optional)",
                        placeholder="e.g. CELL_001",
                        scale=2,
                    )
                    kpi_metric = gr.Dropdown(
                        choices=["All KPIs", "rsrp", "sinr", "rrc_success_rate"],
                        value="All KPIs",
                        label="KPI Metric",
                        scale=1,
                    )
                    kpi_top_n = gr.Slider(
                        minimum=3, maximum=10, value=5, step=1,
                        label="Top N anomalies",
                        scale=1,
                    )
                kpi_btn = gr.Button("📈 Detect Anomalies", variant="primary")
                kpi_output = gr.Markdown("_Click 'Detect Anomalies' to run KPI analysis._")
                kpi_btn.click(
                    fn=run_kpi_analysis,
                    inputs=[kpi_cell, kpi_metric, kpi_top_n],
                    outputs=kpi_output,
                )

            # ──────────── TAB 4: About ───────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## TeleRAG-Agent

                **An agentic AI assistant for telecom network management.**

                ### What This System Does
                - **Answers 3GPP specification questions** using hybrid retrieval from 15+ spec documents
                - **Diagnoses network faults** using real O-RAN testbed data + alarm correlation
                - **Detects KPI anomalies** using statistical z-score analysis (TS 28.552 aligned)
                - **Self-reflects**: if confidence is low, it retries retrieval or asks for clarification

                ### Technical Architecture
                | Component | Technology |
                |---|---|
                | LLM | LLaMA-3 8B (Tele-it) + QLoRA fine-tuned adapter |
                | Retrieval | Hybrid: BGE-large dense + BM25 sparse + KG, RRF fusion |
                | Re-ranking | BGE-reranker-v2-m3 cross-encoder |
                | Agent | LangGraph PLAN → RETRIEVE → GENERATE → REFLECT |
                | Vector DB | Qdrant (self-hosted) |
                | O-RAN Data | NetsLab-5GORAN-IDD + synthetic alarm simulation |
                | Security | Input sanitizer + PII redactor + audit logger |

                ### Models Used
                - `AliMaatouk/LLama-3-8B-Tele-it` — base telecom LLM
                - `BAAI/bge-large-en-v1.5` — dense embeddings
                - `BAAI/bge-reranker-v2-m3` — cross-encoder re-ranking
                - `chaitanyakadupukutla/TeleRAG-LoRA` — our fine-tuned LoRA adapter

                ### Datasets Used
                - TeleQnA (10K telecom Q&A pairs)
                - 3GPP Technical Specifications (TS 38.331, 38.300, 38.321, etc.)
                - NetsLab-5GORAN-IDD (real O-RAN testbed measurements)

                ---
                *Samsung EnnovateX AX Hackathon 2024 — Problem 10: RAG-based Telecom RAN Assistant*  
                *Team: under_served | IIIT Hyderabad*
                """)

        # ── Footer ────────────────────────────────────────────
        gr.HTML("""
        <div style='text-align:center; color:#64748b; font-size:0.8rem; margin-top:16px;'>
            TeleRAG-Agent • Samsung EnnovateX AX Hackathon • 
            <a href='https://github.com/imchaitanya0/TeleRAG-Agent' style='color:#60a5fa;'>GitHub</a>
        </div>
        """)

    return demo


# ── Entry point ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TeleRAG-Agent Gradio UI")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(max_size=10)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
        show_api=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
        ),
    )


if __name__ == "__main__":
    main()
