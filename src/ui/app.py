"""
src/ui/app.py

TeleRAG-Agent: Gradio Chat Interface (Gradio 6.x compatible)

Features:
  - Natural language chat with PLAN/RETRIEVE/GENERATE/REFLECT agent
  - Thinking trace accordion showing all agent reasoning steps
  - Source citation panel with relevance scores
  - Confidence indicator (color-coded)
  - O-RAN Alarm Analysis tab (real NetsLab data + synthetic)
  - KPI Anomaly Detection tab (z-score, TS 28.552 aligned)
  - Security: prompt injection detection + PII redaction + audit logging

Launch:
    source .venv/bin/activate
    python src/ui/app.py

Kaggle / share:
    python src/ui/app.py --share
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

# ── module-level audit logger ──────────────────────────────────
_audit = get_audit_logger()

# ── example queries ────────────────────────────────────────────
EXAMPLE_QUERIES = [
    ["What is RRC Connection Reconfiguration in 5G NR?"],
    ["Explain DRX operation and its parameters in LTE."],
    ["What is the role of AMF in the 5G core network?"],
    ["Why is my cell experiencing high handover failure rate?"],
    ["How do I optimize RSRP thresholds to reduce unnecessary handovers?"],
    ["What is HARQ and how does it improve reliability in NR?"],
    ["What are the 3GPP specifications for O-RAN architecture?"],
]

# ── CSS injected via gr.HTML <style> block ─────────────────────
# (Gradio 6 dropped css= from both Blocks() and launch())
STYLE_HTML = """
<style>
  /* ── Global typography ── */
  body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  }

  /* ── Header gradient card ── */
  .trag-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 12px;
    border: 1px solid rgba(99,179,237,0.25);
  }
  .trag-header h1 {
    color: #e2e8f0;
    font-size: 1.9rem;
    margin: 0;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  .trag-header p {
    color: #94a3b8;
    margin: 6px 0 0;
    font-size: 0.95rem;
  }

  /* ── Source table compact ── */
  .src-table table { font-size: 0.82rem !important; }

  /* ── Thinking trace box ── */
  .think-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
  }
</style>
"""


# ─────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────

def chat(user_message: str, history: list):
    """
    Generator: yields (history, thinking_md, sources_rows, conf_text, type_text)
    on each update so the UI streams changes in real time.
    """
    if not user_message or not user_message.strip():
        yield history, "_Empty query — please ask a question._", [], "—", "—"
        return

    # 1 ── Security: sanitize ────────────────────────────────
    session_id = _audit.start_session(user_message)
    try:
        clean_query = sanitize_input(user_message)
    except InputSanitizationError as exc:
        _audit.log_security_event(session_id, "query_rejected", {"reason": str(exc)})
        _audit.log_completion(session_id, confidence=0.0, iterations=0)
        err = f"⚠️ **Security check failed:** {exc}"
        yield (
            history + [[user_message, err]],
            "_Query blocked by security filter._",
            [], "0%", "🔒 Blocked",
        )
        return

    # 2 ── Show immediate "thinking" placeholder ──────────────
    thinking_placeholder = (
        "⏳ _Running agent…_\n\n"
        "**PLAN** → **RETRIEVE** → **GENERATE** → **REFLECT**"
    )
    yield (
        history + [[user_message, "⏳ _Thinking…_"]],
        thinking_placeholder,
        [],
        "—",
        "—",
    )

    # 3 ── Run the LangGraph agent ────────────────────────────
    t0 = time.perf_counter()
    try:
        result = run_agent(clean_query)
    except Exception as exc:
        _audit.log_error(session_id, str(exc))
        _audit.log_completion(session_id, confidence=0.0, iterations=0)
        err = f"❌ **Agent error:** {str(exc)[:300]}"
        yield (
            history + [[user_message, err]],
            f"_Error: {exc}_",
            [], "0%", "❌",
        )
        return

    latency_ms = (time.perf_counter() - t0) * 1000

    # 4 ── PII redaction on output ────────────────────────────
    raw_answer = result.get("final_answer", "No answer generated.")
    pii = detect_and_redact(raw_answer)
    if pii.has_pii:
        _audit.log_security_event(session_id, "pii_redacted",
                                  {"types": [f["name"] for f in pii.findings]})
    clean_answer = pii.redacted_text

    # 5 ── Build outputs ──────────────────────────────────────
    confidence   = result.get("confidence", 0.0)
    iterations   = result.get("iteration", 1)
    query_type   = result.get("query_type", "")
    sources      = result.get("sources", [])

    footer = (
        f"\n\n---\n"
        f"_⏱️ {latency_ms:.0f} ms &nbsp;|&nbsp; "
        f"🔁 {iterations} iteration(s) &nbsp;|&nbsp; "
        f"📚 {len(sources)} source(s)_"
    )
    full_answer = clean_answer + footer

    thinking_md   = format_thinking_trace({
        **result,
        "sub_queries_used": result.get("sub_queries", [clean_query]),
    })
    sources_rows  = format_sources_table(sources)
    conf_text     = confidence_label(confidence)
    type_text     = query_type_badge(query_type)

    # 6 ── Audit ──────────────────────────────────────────────
    _audit.log_completion(
        session_id,
        confidence=confidence,
        iterations=iterations,
        query_type=query_type,
        sources_used=len(sources),
        answer_length=len(clean_answer),
    )

    yield (
        history + [[user_message, full_answer]],
        thinking_md,
        sources_rows,
        conf_text,
        type_text,
    )


def run_alarm_analysis(cell_id: str, severity: str, top_n: int) -> str:
    cell = cell_id.strip() or None
    sev  = severity if severity != "All" else None
    return format_alarm_result(alarm_analyzer_tool(cell_id=cell, severity=sev, top_n=int(top_n)))


def run_kpi_analysis(cell_id: str, kpi_name: str, top_n: int) -> str:
    cell = cell_id.strip() or None
    kpi  = kpi_name if kpi_name != "All KPIs" else None
    return format_kpi_result(kpi_calculator_tool(cell_id=cell, kpi_name=kpi, top_n=int(top_n)))


# ─────────────────────────────────────────────────────────────
# UI definition
# ─────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TeleRAG-Agent — 5G Telecom AI Assistant") as demo:

        # Inject CSS via HTML style block (Gradio 6 compatible)
        gr.HTML(STYLE_HTML)

        # ── Header ─────────────────────────────────────────────
        gr.HTML("""
        <div class="trag-header">
          <h1>📡 TeleRAG-Agent</h1>
          <p>AI assistant for 3GPP specifications · O-RAN fault analysis · Network KPI monitoring</p>
        </div>
        """)

        with gr.Tabs():

            # ══════════════ TAB 1: CHAT ═══════════════════════
            with gr.TabItem("💬 Ask a Question"):

                with gr.Row(equal_height=False):

                    # Left: chat + input
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="TeleRAG-Agent",
                            height=460,
                            show_label=False,
                            layout="bubble",
                        )

                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="Ask about 5G NR specs, O-RAN faults, KPIs…",
                                show_label=False,
                                scale=5,
                                container=False,
                                max_lines=3,
                                autofocus=True,
                            )
                            send_btn = gr.Button("Send ➤", variant="primary",
                                                 scale=1, min_width=80)

                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

                        # Example queries
                        gr.Markdown("**Quick examples:**")
                        gr.Examples(
                            examples=EXAMPLE_QUERIES,
                            inputs=[msg_box],
                            label="",
                        )

                    # Right: confidence + type + sources
                    with gr.Column(scale=1, min_width=240):
                        conf_out = gr.Markdown("**Confidence:** —")
                        type_out = gr.Markdown("**Query type:** —")

                        gr.Markdown("---\n### 📚 Sources")
                        sources_out = gr.Dataframe(
                            headers=SOURCES_HEADERS,
                            datatype=["number", "str", "str", "str", "str"],
                            label=None,
                            interactive=False,
                            wrap=True,
                            elem_classes=["src-table"],
                            max_height=260,
                        )

                # Thinking trace accordion (below both columns)
                with gr.Accordion("🧠 Agent Thinking Trace", open=False):
                    thinking_out = gr.Markdown(
                        "_Ask a question to see the agent's step-by-step reasoning here._",
                        elem_classes=["think-box"],
                    )

                # ── Event wiring ──────────────────────────────
                _ins  = [msg_box, chatbot]
                _outs = [chatbot, thinking_out, sources_out, conf_out, type_out]

                send_btn.click(fn=chat, inputs=_ins, outputs=_outs).then(
                    fn=lambda: "", inputs=None, outputs=msg_box
                )
                msg_box.submit(fn=chat, inputs=_ins, outputs=_outs).then(
                    fn=lambda: "", inputs=None, outputs=msg_box
                )
                clear_btn.click(
                    fn=lambda: ([], "_Cleared._", [], "—", "—"),
                    inputs=None,
                    outputs=_outs,
                )

            # ══════════════ TAB 2: ALARMS ══════════════════════
            with gr.TabItem("🚨 O-RAN Alarm Analysis"):
                gr.Markdown("""
### Real O-RAN Network Fault Analysis
Analyses alarm events from the **NetsLab-5GORAN-IDD** real testbed dataset
and synthetic O-RAN alarm logs. Detects alarm storms and identifies probable root causes.
                """)
                with gr.Row():
                    alarm_cell = gr.Textbox(
                        label="Cell ID (optional)",
                        placeholder="e.g. CELL_001",
                        scale=2,
                    )
                    alarm_sev = gr.Dropdown(
                        choices=["All", "critical", "major", "minor", "warning"],
                        value="All",
                        label="Severity",
                        scale=1,
                    )
                    alarm_n = gr.Slider(3, 10, value=5, step=1,
                                        label="Top N", scale=1)

                alarm_btn = gr.Button("🔍 Analyse Alarms", variant="primary")
                alarm_out = gr.Markdown("_Click the button to run alarm analysis._")
                alarm_btn.click(
                    fn=run_alarm_analysis,
                    inputs=[alarm_cell, alarm_sev, alarm_n],
                    outputs=alarm_out,
                )

            # ══════════════ TAB 3: KPI ═════════════════════════
            with gr.TabItem("📊 KPI Anomaly Detection"):
                gr.Markdown("""
### Network KPI Anomaly Detection
Z-score statistical analysis on cell-level KPI time-series.
Detects **RSRP** degradation, **SINR** drops, and **RRC Success Rate** anomalies
aligned with **3GPP TS 28.552** thresholds.
                """)
                with gr.Row():
                    kpi_cell = gr.Textbox(
                        label="Cell ID (optional)",
                        placeholder="e.g. CELL_003",
                        scale=2,
                    )
                    kpi_metric = gr.Dropdown(
                        choices=["All KPIs", "rsrp", "sinr", "rrc_success_rate"],
                        value="All KPIs",
                        label="KPI",
                        scale=1,
                    )
                    kpi_n = gr.Slider(3, 10, value=5, step=1,
                                      label="Top N anomalies", scale=1)

                kpi_btn = gr.Button("📈 Detect Anomalies", variant="primary")
                kpi_out = gr.Markdown("_Click the button to run KPI analysis._")
                kpi_btn.click(
                    fn=run_kpi_analysis,
                    inputs=[kpi_cell, kpi_metric, kpi_n],
                    outputs=kpi_out,
                )

            # ══════════════ TAB 4: ABOUT ════════════════════════
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
## TeleRAG-Agent

**An agentic AI assistant for 5G telecom network management.**

### What This System Does
- **Answers 3GPP specification questions** — hybrid retrieval from 15+ spec documents (TS 38.331, 38.300, 38.321…)
- **Diagnoses network faults** — real O-RAN testbed data + temporal alarm correlation
- **Detects KPI anomalies** — z-score statistics aligned with TS 28.552 thresholds
- **Self-reflects** — if confidence is low, the agent re-retrieves or asks a clarifying question

### Technical Architecture

| Component | Technology |
|---|---|
| LLM | LLaMA-3 8B Tele-it + QLoRA fine-tuned LoRA adapter |
| Retrieval | BGE-large dense + BM25 sparse + KG heading index |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Re-ranking | BGE-reranker-v2-m3 cross-encoder |
| Agent | LangGraph: PLAN → RETRIEVE → GENERATE → REFLECT |
| Vector DB | Qdrant (self-hosted, local path) |
| O-RAN Data | NetsLab-5GORAN-IDD + synthetic alarm simulation |
| Security | Input sanitizer · PII redactor · JSON audit logger |

### Models
- [`AliMaatouk/LLama-3-8B-Tele-it`](https://huggingface.co/AliMaatouk/LLama-3-8B-Tele-it) — base telecom LLM
- [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) — dense embeddings
- [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) — cross-encoder re-ranking
- [`chaitanyakadupukutla/TeleRAG-LoRA`](https://huggingface.co/chaitanyakadupukutla/TeleRAG-LoRA) — our fine-tuned adapter

### Datasets
- [TeleQnA](https://huggingface.co/datasets/netop-team/TeleQnA) — 10K telecom Q&A pairs
- [3GPP R16/R18 Specifications](https://www.3gpp.org/specifications) — 15+ technical specs
- [NetsLab-5GORAN-IDD](https://www.kaggle.com/datasets/netslab/5goran-idd) — real O-RAN testbed measurements

---
*Samsung EnnovateX AX Hackathon · Problem 10: RAG-based Future-Ready Telecom RAN Assistant*  
*Team: under_served · IIIT Hyderabad, Gachibowli*
                """)

        # ── Footer ─────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; color:#64748b; font-size:0.8rem; margin-top:16px; padding-bottom:8px;">
          TeleRAG-Agent &nbsp;·&nbsp; Samsung EnnovateX AX Hackathon &nbsp;·&nbsp;
          <a href="https://github.com/imchaitanya0/TeleRAG-Agent"
             style="color:#60a5fa; text-decoration:none;">GitHub ↗</a>
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TeleRAG-Agent Gradio UI")
    parser.add_argument("--port",   type=int,  default=7860,      dest="server_port")
    parser.add_argument("--host",   type=str,  default="0.0.0.0", dest="server_name")
    parser.add_argument("--share",  action="store_true")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(max_size=10)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
