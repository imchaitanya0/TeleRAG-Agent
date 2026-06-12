"""
scripts/test_pipeline.py

Verification suite for Tasks 1, 2, and 3.

Usage:
    # Fast tests (prompt format + retrieval only — no LLM load):
    python scripts/test_pipeline.py --fast

    # Full tests (loads the 8B model — slow on CPU, ~5-10 min):
    python scripts/test_pipeline.py

What to look for:
  - Each test prints [PASS] or [FAIL] with details.
  - "All tests passed!" means everything is wired correctly.
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


# ─────────────────────────────────────────────
# Test 1: Model Loader
# ─────────────────────────────────────────────
def test_loader(fast: bool = False) -> bool:
    print("\n" + "="*60)
    print("TEST 1: Model Loader (src/models/loader.py)")
    print("="*60)

    if fast:
        # Just verify the module imports and the singleton logic is present
        try:
            from src.models.loader import get_model, reset_model, HAS_CUDA
            print(f"{PASS} Module imports correctly")
            print(f"{INFO} CUDA available: {HAS_CUDA}")
            print(f"{SKIP} Skipping actual model load (--fast mode). Run without --fast to load the 8B model.")
            return True
        except Exception as e:
            print(f"{FAIL} Import error: {e}")
            return False

    try:
        from src.models.loader import get_model, reset_model

        t0 = time.perf_counter()
        print("[Loader] Loading model (this may take several minutes on CPU)...")
        model, tokenizer = get_model(load_in_4bit=False, lora_repo=None)
        elapsed = (time.perf_counter() - t0)

        assert model is not None, "model is None"
        assert tokenizer is not None, "tokenizer is None"
        assert tokenizer.pad_token is not None, "pad_token is None"
        print(f"{PASS} Model loaded: {type(model).__name__} in {elapsed:.1f}s")
        print(f"{PASS} Device: {next(model.parameters()).device}")
        print(f"{PASS} Tokenizer pad token: '{tokenizer.pad_token}'")

        # Singleton check
        model2, _ = get_model()
        assert model is model2, "Singleton broken — returned different instance"
        print(f"{PASS} Singleton: second call returned cached instance instantly")
        return True

    except Exception as e:
        print(f"{FAIL} {e}")
        import traceback; traceback.print_exc()
        return False


# ─────────────────────────────────────────────
# Test 2: Inference Module
# ─────────────────────────────────────────────
def test_inference(fast: bool = False) -> bool:
    print("\n" + "="*60)
    print("TEST 2: Inference Module (src/models/inference.py)")
    print("="*60)
    try:
        from src.models.inference import (
            build_mcq_prompt, build_open_prompt,
            extract_letter, generate,
        )

        # ── Prompt format (always run, no model needed) ──────────
        mcq_prompt = build_mcq_prompt(
            question="What is DRX?",
            options=["Discontinuous Reception", "Dynamic Resource eXchange",
                     "Data Relay Extension", "None"],
        )
        assert "### Question:" in mcq_prompt, "Missing '### Question:' header"
        assert "### Answer:" in mcq_prompt, "Missing '### Answer:' trigger"
        assert "A) Discontinuous Reception" in mcq_prompt, "Option A missing"
        assert "D) None" in mcq_prompt, "Option D missing"
        print(f"{PASS} build_mcq_prompt() format is correct")
        print(f"{INFO} Prompt preview:\n{mcq_prompt}\n")

        open_prompt = build_open_prompt(
            "What is 5G NR?",
            context="5G NR (New Radio) is the global standard for a new generation of cellular networks. " * 5  # >100 chars
        )
        assert "### Question:" in open_prompt
        assert "Relevant information:" in open_prompt
        print(f"{PASS} build_open_prompt() with context is correct")

        # ── extract_letter (always run, no model needed) ─────────
        cases = [
            ("The answer is: A", 4, "A"),
            ("The answer is: C", 4, "C"),
            ("B is the correct answer", 4, "B"),
            ("D", 4, "D"),
        ]
        for raw, n, expected in cases:
            letter = extract_letter(raw, num_options=n)
            assert letter == expected, f"Expected {expected}, got {letter} for '{raw}'"
        print(f"{PASS} extract_letter() correctly parses {len(cases)} mock responses")

        if fast:
            print(f"{SKIP} Skipping generate() (--fast mode). Run without --fast to test generation.")
            return True

        # ── generation (requires model) ──────────────────────────
        t0 = time.perf_counter()
        response = generate(mcq_prompt, max_new_tokens=20, lora_repo=None)
        elapsed = (time.perf_counter() - t0) * 1000
        assert isinstance(response, str), f"generate() returned {type(response)}"
        assert len(response) > 0, "generate() returned empty string"
        print(f"{PASS} generate() returned: '{response}' ({elapsed:.0f}ms)")
        return True

    except Exception as e:
        print(f"{FAIL} {e}")
        import traceback; traceback.print_exc()
        return False


# ─────────────────────────────────────────────
# Test 3: RAG Pipeline
# ─────────────────────────────────────────────
def test_rag_pipeline(fast: bool = False) -> bool:
    print("\n" + "="*60)
    print("TEST 3: RAG Pipeline (src/pipeline/rag_pipeline.py)")
    print("="*60)
    try:
        # ── Retrieval-only check (always run) ────────────────────
        from src.retrieval.fusion import HybridRetriever
        from src.retrieval.reranker import Reranker
        from src.retrieval.context_assembler import ContextAssembler

        query = "What is RRC Connection Reconfiguration in 5G NR?"
        print(f"{INFO} Query: {query}")

        retriever = HybridRetriever()
        candidates = retriever.search(query, top_k=10)
        assert len(candidates) > 0, "Retriever returned 0 results"
        print(f"{PASS} HybridRetriever returned {len(candidates)} candidates")

        reranker = Reranker()
        reranked = reranker.rerank(query, candidates, top_k=5, threshold=0.0)
        assert len(reranked) > 0, "Reranker returned 0 results"
        print(f"{PASS} Reranker returned {len(reranked)} reranked results")
        print(f"   Top source: {reranked[0]['payload'].get('spec_number')} "
              f"§{reranked[0]['payload'].get('clause_string')} "
              f"[score: {reranked[0].get('rerank_score', 0):.4f}]")

        assembler = ContextAssembler()
        context = assembler.assemble(reranked)
        assert isinstance(context, str) and len(context) > 50, "Context is too short"
        print(f"{PASS} ContextAssembler produced {len(context)} chars of context")
        print(f"   First 200 chars: {context[:200]}...")

        if fast:
            print(f"{SKIP} Skipping LLM generation in pipeline (--fast mode).")
            return True

        # ── Full pipeline with LLM ───────────────────────────────
        from src.pipeline.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(lora_repo=None)

        t0 = time.perf_counter()
        result = pipeline.run(query, max_new_tokens=100)
        elapsed = (time.perf_counter() - t0) * 1000

        assert "answer" in result
        assert "context" in result
        assert "sources" in result
        assert isinstance(result["answer"], str) and len(result["answer"]) > 0

        print(f"{PASS} Full pipeline returned valid result in {elapsed:.0f}ms")
        print(f"\n--- Answer ---\n{result['answer'][:300]}")
        print(f"\n--- Sources ({len(result['sources'])} total) ---")
        for s in result["sources"][:3]:
            print(f"  {s['spec']} §{s['clause']}  [{s['score']}]  {s['title']}")

        if elapsed < 30_000:
            print(f"{PASS} Latency {elapsed:.0f}ms OK")
        else:
            print(f"{WARN} Latency {elapsed:.0f}ms is high (expected on CPU)")

        return True

    except Exception as e:
        print(f"{FAIL} {e}")
        import traceback; traceback.print_exc()
        return False


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip model loading — verify prompt format and retrieval only (runs in ~30s)."
    )
    args = parser.parse_args()

    mode = "FAST (retrieval + format checks only)" if args.fast else "FULL (includes model load)"
    print(f"\n{'='*60}")
    print(f"TeleRAG-Agent Pipeline Tests  [{mode}]")
    print(f"{'='*60}")

    results = [
        test_loader(fast=args.fast),
        test_inference(fast=args.fast),
        test_rag_pipeline(fast=args.fast),
    ]

    print("\n" + "="*60)
    if all(results):
        print("\033[92mAll tests passed!\033[0m")
        if args.fast:
            print("Run without --fast to fully verify model loading and generation.")
    else:
        failed = [i+1 for i, r in enumerate(results) if not r]
        print(f"\033[91mFailed tests: {failed}\033[0m")
        sys.exit(1)
