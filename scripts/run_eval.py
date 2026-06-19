"""
scripts/run_eval.py

CLI runner for TeleRAG-Agent evaluation framework.

Modes:
  --mode retrieval  : MRR@10, Recall@k (no LLM, fast)
  --mode accuracy   : MCQ accuracy on golden questions (needs LLM)
  --mode ablation   : All 4 ablation experiments (needs LLM, slow)

Usage on Kaggle (after kaggle_setup.py):
    # Fast retrieval metrics only (no LLM needed):
    python scripts/run_eval.py --mode retrieval --n 100

    # Full ablation (needs GPU, ~30-60 min):
    python scripts/run_eval.py --mode ablation --n 50

    # MCQ accuracy with LoRA:
    python scripts/run_eval.py --mode accuracy --n 100 --lora

Local usage:
    QDRANT_URL="" python scripts/run_eval.py --mode retrieval --n 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on path
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(str(REPO_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="TeleRAG-Agent Evaluation Runner")
parser.add_argument(
    "--mode", choices=["retrieval", "accuracy", "ablation"],
    required=True, help="Which evaluation to run"
)
parser.add_argument("--n", type=int, default=50, help="Number of questions (default 50)")
parser.add_argument("--lora", action="store_true", help="Load LoRA adapter for generation")
parser.add_argument("--verbose", action="store_true", help="Print per-question results")
parser.add_argument(
    "--out-dir", type=str, default="eval_results",
    help="Directory to save results (default: eval_results/)"
)
args = parser.parse_args()

OUT_DIR = REPO_DIR / args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

def banner(msg):
    print(f"\n{'═'*60}\n  {msg}\n{'═'*60}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load golden questions from TeleQnA test set
# ─────────────────────────────────────────────────────────────────────────────
def load_golden_questions(n: int) -> list[dict]:
    """
    Load up to n questions from the TeleQnA test set.
    Falls back to val set if test set is smaller than n.
    """
    from src.config import DATA_PROCESSED_DIR
    import json

    # Try different file paths
    candidates = [
        DATA_PROCESSED_DIR / "teleqna_test.jsonl",
        DATA_PROCESSED_DIR / "teleqna_val.jsonl",
        REPO_DIR / "data" / "processed" / "teleqna_test.jsonl",
        REPO_DIR / "data" / "processed" / "teleqna_val.jsonl",
    ]

    questions = []
    for path in candidates:
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Normalize to our eval format
                    q = {
                        "question": raw.get("question", raw.get("instruction", "")),
                        "options":  raw.get("options", []),
                        "answer":   raw.get("answer", raw.get("output", "")),
                        "gold_specs": raw.get("gold_specs", []),
                    }

                    # Normalize answer to letter (A/B/C/D/E)
                    ans = q["answer"].strip()
                    if ans and ans[0].isdigit():
                        # Convert "1" → "A", "2" → "B", etc.
                        idx = int(ans[0]) - 1
                        if 0 <= idx <= 4:
                            q["answer"] = chr(65 + idx)

                    if q["question"] and q["options"] and q["answer"]:
                        questions.append(q)

            print(f"  Loaded {len(questions)} questions from {path.name}")
            break

    if not questions:
        print("  ⚠ No golden questions found. Generating synthetic test questions...")
        questions = _synthetic_questions()

    return questions[:n]


def _synthetic_questions() -> list[dict]:
    """Fallback: basic telecom questions for smoke-testing."""
    return [
        {
            "question": "What is the primary function of the RRC protocol in 5G NR?",
            "options": [
                "Radio Resource Control for connection management",
                "Radio Frequency Control for spectrum allocation",
                "Remote Radio Control for antenna management",
                "Rapid Resource Creation for new connections",
            ],
            "answer": "A",
            "gold_specs": ["TS 38.331"],
        },
        {
            "question": "Which layer is responsible for HARQ in 5G NR?",
            "options": [
                "RRC", "PDCP", "RLC", "MAC"
            ],
            "answer": "D",
            "gold_specs": ["TS 38.321"],
        },
        {
            "question": "What does DRX stand for in LTE/NR?",
            "options": [
                "Dynamic Radio Exchange",
                "Discontinuous Reception",
                "Digital Radio Extension",
                "Dual Radio Excitation",
            ],
            "answer": "B",
            "gold_specs": ["TS 38.321", "TS 36.321"],
        },
        {
            "question": "What is the maximum number of HARQ processes supported in NR DL?",
            "options": ["8", "16", "32", "4"],
            "answer": "B",
            "gold_specs": ["TS 38.321"],
        },
        {
            "question": "In 5G NR, what is the role of the AMF in the core network?",
            "options": [
                "Access and Mobility Management Function",
                "Application Media Framework",
                "Advanced Modulation Function",
                "Antenna Management Framework",
            ],
            "answer": "A",
            "gold_specs": ["TS 23.501"],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# RUN SELECTED MODE
# ─────────────────────────────────────────────────────────────────────────────
banner(f"TeleRAG-Agent Evaluation — mode={args.mode}, n={args.n}")

questions = load_golden_questions(args.n)
print(f"  Using {len(questions)} questions\n")

timestamp = time.strftime("%Y%m%d_%H%M%S")

if args.mode == "retrieval":
    banner("Retrieval Metrics (MRR@10, Recall@1/5/10)")
    from src.evaluation.retrieval_metrics import evaluate_retrieval

    results = evaluate_retrieval(
        questions, top_k=10, rerank_k=5, use_reranker=True, verbose=args.verbose
    )
    results_no_rerank = evaluate_retrieval(
        questions, top_k=10, rerank_k=5, use_reranker=False, verbose=False
    )

    print(f"\n  Results WITH reranker:")
    print(f"    MRR@10:     {results['mrr_at_k']:.4f}")
    print(f"    Recall@1:   {results['recall_at_1']*100:.1f}%")
    print(f"    Recall@5:   {results['recall_at_5']*100:.1f}%")
    print(f"    Recall@10:  {results['recall_at_k']*100:.1f}%")
    print(f"    Avg latency:{results['avg_latency_ms']:.0f}ms")

    print(f"\n  Results WITHOUT reranker:")
    print(f"    MRR@10:     {results_no_rerank['mrr_at_k']:.4f}")
    print(f"    Recall@5:   {results_no_rerank['recall_at_5']*100:.1f}%")

    out = {
        "mode": "retrieval",
        "n_questions": len(questions),
        "with_reranker": results,
        "without_reranker": results_no_rerank,
    }
    out_path = OUT_DIR / f"retrieval_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  💾 Saved to {out_path}")

elif args.mode == "accuracy":
    banner("MCQ Answer Accuracy")
    from src.evaluation.answer_metrics import evaluate_answers
    from src.config import LORA_REPO

    lora = LORA_REPO if args.lora else None
    print(f"  LoRA: {lora or 'none (base model)'}")

    results = evaluate_answers(
        questions, use_rag=True, lora_repo=lora,
        max_new_tokens=50, verbose=args.verbose
    )

    print(f"\n  Results:")
    print(f"    Accuracy:        {results['accuracy']*100:.1f}%  ({results['n_correct']}/{results['n_questions']-results['n_abstained']} answered)")
    print(f"    Exact Match:     {results['exact_match']*100:.1f}%  (incl. abstentions)")
    print(f"    Abstention Rate: {results['abstention_rate']*100:.1f}%")
    print(f"    Avg latency:     {results['avg_latency_ms']:.0f}ms")

    out = {"mode": "accuracy", "lora": lora, **results}
    out_path = OUT_DIR / f"accuracy_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  💾 Saved to {out_path}")

elif args.mode == "ablation":
    banner("Ablation Study (4 experiments)")
    from src.evaluation.ablation import run_ablation, print_ablation_table

    results = run_ablation(
        questions, n=len(questions), verbose=True,
        save_path=OUT_DIR / f"ablation_{timestamp}.json"
    )

    print_ablation_table(results)
    print(f"\n  💾 Saved to {OUT_DIR}/ablation_{timestamp}.json")

print(f"\n✅ Evaluation complete.")
