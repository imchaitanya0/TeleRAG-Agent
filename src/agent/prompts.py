"""
src/agent/prompts.py

All LLM prompt templates for the TeleRAG-Agent agentic loop.

Centralising prompts here makes them easy to iterate without touching node logic.
Every prompt uses the same ### Question: / ### Answer: format the model was trained on.
"""

# ──────────────────────────────────────────────────────────────
# PLAN node — query classification + decomposition
# ──────────────────────────────────────────────────────────────

CLASSIFICATION_PROMPT = """### Question:
You are a telecom expert assistant. Classify the following user query into exactly one category and list up to 3 focused sub-queries for retrieval.

Categories:
- spec_qa       : Question about a specific 3GPP specification or standard
- troubleshoot  : Root cause analysis or fault diagnosis query
- optimization  : Network parameter tuning or optimization query
- kpi           : KPI anomaly, threshold, or measurement query
- general       : General telecom concept explanation

User query: {query}

Respond in this EXACT format (no extra text):
CATEGORY: <category>
SUB_QUERIES:
1. <sub-query 1>
2. <sub-query 2>
3. <sub-query 3>

### Answer:
"""

# ──────────────────────────────────────────────────────────────
# GENERATE node — grounded answer generation
# ──────────────────────────────────────────────────────────────

GENERATION_PROMPT = """### Question:
You are a telecom domain expert assistant specialising in 3GPP standards and O-RAN systems.

Answer the following query using ONLY the provided reference material.
Be precise and concise. After your answer, list the sources you used.

Reference material:
{context}

Query: {query}

### Answer:
"""

GENERATION_PROMPT_NO_CONTEXT = """### Question:
You are a telecom domain expert assistant specialising in 3GPP standards and O-RAN systems.

Answer the following query based on your knowledge. Acknowledge if you are uncertain.

Query: {query}

### Answer:
"""

# ──────────────────────────────────────────────────────────────
# REFLECT node — self-critique + confidence
# ──────────────────────────────────────────────────────────────

REFLECTION_PROMPT = """### Question:
You are a quality reviewer for a telecom AI assistant.

Evaluate the following answer against the original query and reference material.

Original query: {query}

Reference material (excerpt):
{context_excerpt}

Generated answer: {answer}

Rate the answer on:
1. Accuracy (does it correctly answer the query based on the reference?)
2. Completeness (does it cover all aspects of the query?)
3. Groundedness (is every claim supported by the reference material?)

Respond in this EXACT format:
CONFIDENCE: <0.0-1.0>
GAPS: <what is missing or incorrect, or "None">
VERDICT: <"sufficient" | "needs_retrieval" | "needs_clarification">

### Answer:
"""

# ──────────────────────────────────────────────────────────────
# Clarifying question template (not an LLM call — string format)
# ──────────────────────────────────────────────────────────────

CLARIFICATION_TEMPLATE = (
    "I need a bit more context to give you an accurate answer.\n\n"
    "Could you clarify:\n"
    "{gaps}\n\n"
    "For example, are you asking about a specific 3GPP release, "
    "a particular network configuration, or a specific alarm type?"
)
