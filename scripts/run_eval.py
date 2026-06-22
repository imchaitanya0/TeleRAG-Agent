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
parser.add_argument("--dataset", type=str, default="test", choices=["test", "val", "standards"], help="Dataset split to evaluate on")
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
# Load golden questions from TeleQnA dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_golden_questions(n: int, dataset_name: str = "test") -> list[dict]:
    """
    Load up to n questions for evaluation.

    The teleqna_test.jsonl format:
        {"instruction": "...", "input": "Question: ...\n\nOptions:\nA) ...", "output": "The answer is: E"}

    We parse this into:
        {"question": "...", "options": ["opt1", ...], "answer": "E",
         "input_text": "<raw input field for exact prompt match>"}
    """
    import json
    import re
    from pathlib import Path
    from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

    questions = []

    # ── 1: Search for processed JSONL splits ──────────────────────
    filename = f"teleqna_{dataset_name}.jsonl"
    search_paths = [
        DATA_PROCESSED_DIR / filename,
    ]

    # Also search Kaggle inputs for pre-processed jsonl datasets
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        search_paths.extend(list(kaggle_input.rglob(filename)))

    for path in search_paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # ── Parse the training-format fields ──
                input_text = raw.get("input", "")
                output_text = raw.get("output", "")

                # Extract gold letter from "The answer is: E"
                gold_match = re.search(r"[Tt]he\s+answer\s+is[:\s]+([A-E])", output_text)
                if gold_match:
                    gold = gold_match.group(1).upper()
                elif output_text.strip() and output_text.strip()[0] in "ABCDE":
                    gold = output_text.strip()[0].upper()
                else:
                    continue  # skip if we can't determine the answer

                # Extract question text from input (strip "Question: " prefix)
                q_match = re.match(r"Question:\s*(.+?)(?:\n\n|\nOptions:)", input_text, re.DOTALL)
                q_text = q_match.group(1).strip() if q_match else input_text.split("\n")[0]

                # Extract individual options — simple line-by-line (captures ALL options including E)
                options = []
                for opt_line in input_text.split("\n"):
                    opt_m = re.match(r"^([A-E])\)\s*(.+)$", opt_line.strip())
                    if opt_m:
                        options.append(opt_m.group(2).strip())


                if not input_text:
                    continue

                questions.append({
                    "question": q_text,
                    "options": options,
                    "answer": gold,
                    "input_text": input_text,   # RAW training-format input for exact prompt match
                    "gold_specs": raw.get("gold_specs", []),
                })

        if questions:
            print(f"  Loaded {len(questions)} questions from {path.name}")
            break

    # ── 2: Kaggle native TeleQnA.json (raw format) ───────────────
    if not questions:
        raw_paths = []
        if kaggle_input.exists():
            raw_paths.extend(list(kaggle_input.rglob("TeleQnA.json")))
        local_raw = DATA_RAW_DIR / "teleqna" / "TeleQnA.json"
        if local_raw.exists():
            raw_paths.append(local_raw)

        for teleqna_json in raw_paths:
            try:
                with open(teleqna_json) as f:
                    data = json.load(f)
                raw_qs = list(data.values()) if isinstance(data, dict) else data
                for raw in raw_qs:
                    q_text = raw.get("question", "")
                    opts = [raw.get(f"option {i}", "") for i in range(1, 6) if raw.get(f"option {i}")]
                    ans_raw = raw.get("answer", "")
                    letter = ""
                    if ans_raw.startswith("option "):
                        try:
                            idx = int(ans_raw.split()[1]) - 1
                            letter = chr(65 + idx)
                        except (ValueError, IndexError):
                            pass
                    elif ans_raw and ans_raw[0].isalpha():
                        letter = ans_raw[0].upper()

                    if q_text and opts and letter:
                        # Build training-format input_text for prompt consistency
                        opts_str = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(opts))
                        input_text = f"Question: {q_text}\n\nOptions:\n{opts_str}"
                        questions.append({
                            "question": q_text,
                            "options": opts,
                            "answer": letter,
                            "input_text": input_text,
                            "gold_specs": [],
                        })
                if questions:
                    print(f"  Loaded {len(questions)} questions from {teleqna_json}")
                    import random; random.seed(42); random.shuffle(questions)
                    break
            except Exception as e:
                print(f"  ⚠ Failed to load {teleqna_json}: {e}")

    # ── 3: Synthetic fallback ─────────────────────────────────────
    if not questions:
        print("  ⚠ No real TeleQnA data found — using 5 synthetic questions.")
        questions = _synthetic_questions()

    return questions[:n]


def _synthetic_questions() -> list[dict]:
    """Fallback: basic telecom questions for smoke-testing."""
    synth = [
        ("What is the primary function of the RRC protocol in 5G NR?",
         ["Radio Resource Control for connection management",
          "Radio Frequency Control for spectrum allocation",
          "Remote Radio Control for antenna management",
          "Rapid Resource Creation for new connections"],
         "A", ["TS 38.331"]),
        ("Which layer is responsible for HARQ in 5G NR?",
         ["RRC", "PDCP", "RLC", "MAC"], "D", ["TS 38.321"]),
        ("What does DRX stand for in LTE/NR?",
         ["Dynamic Radio Exchange", "Discontinuous Reception",
          "Digital Radio Extension", "Dual Radio Excitation"],
         "B", ["TS 38.321", "TS 36.321"]),
        ("What is the maximum number of HARQ processes supported in NR DL?",
         ["8", "16", "32", "4"], "B", ["TS 38.321"]),
        ("In 5G NR, what is the role of the AMF in the core network?",
         ["Access and Mobility Management Function",
          "Application Media Framework",
          "Advanced Modulation Function",
          "Antenna Management Framework"],
         "A", ["TS 23.501"]),
    ]
    result = []
    for q, opts, ans, specs in synth:
        opts_str = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(opts))
        result.append({
            "question": q, "options": opts, "answer": ans,
            "input_text": f"Question: {q}\n\nOptions:\n{opts_str}",
            "gold_specs": specs,
        })
    return result




# ─────────────────────────────────────────────────────────────────────────────
# RUN SELECTED MODE
# ─────────────────────────────────────────────────────────────────────────────
banner(f"TeleRAG-Agent Evaluation — mode={args.mode}, n={args.n}")

questions = load_golden_questions(args.n, args.dataset)
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
    print(f"    Exact Match:     {results['exact_match']*100:.1f}%  ({results['n_correct']}/{results['n_questions']} total)")
    print(f"    Accuracy*:       {results['accuracy']*100:.1f}%  ({results['n_correct']}/{results['n_questions']-results['n_abstained']} answered only)")
    print(f"    Abstention Rate: {results['abstention_rate']*100:.1f}%  ({results['n_abstained']} questions)")
    print(f"    Avg latency:     {results['avg_latency_ms']:.0f}ms")
    print(f"    * Accuracy excludes abstentions — use Exact Match for fair comparison")

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
