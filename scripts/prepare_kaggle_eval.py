"""
Prepare Kaggle evaluation input by retrieving real RAG context for each TeleQnA question.

This runs locally (uses Qdrant + retrieval pipeline) and produces a JSON file
that gets uploaded to Kaggle for model evaluation.

Usage:
    python scripts/prepare_kaggle_eval.py --num 50 --split val
"""
import json
import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_assembler import ContextAssembler


def parse_teleqna_item(item: dict) -> dict:
    """
    Parse a TeleQnA JSONL item into {question, options, answer}.
    
    Input format:
        instruction: "You are a telecom domain expert..."
        input: "Question: ... [3GPP Release 18]\\n\\nOptions:\\nA) ...\\nB) ...\\nC) ...\\nD) ...\\nE) ..."
        output: "The answer is: option N: ..."
    
    Returns:
        {question, options: [str,...], answer: "A"|"B"|"C"|...}
    """
    input_text = item["input"]
    
    # Split question from options
    parts = re.split(r'\n\s*\n\s*Options:\s*\n', input_text, maxsplit=1)
    question = parts[0].replace("Question: ", "").strip()
    
    # Parse options (A) ... B) ... etc)
    options = []
    if len(parts) > 1:
        options_text = parts[1]
        # Match A) through E) options
        opt_matches = re.findall(r'([A-E])\)\s*(.+?)(?=\n[A-E]\)|$)', options_text, re.DOTALL)
        options = [text.strip() for _, text in opt_matches]
    
    # Parse answer
    output = item.get("output", "")
    answer_letter = "A"  # default
    
    # Try "option N" format
    match = re.search(r'option\s*(\d+)', output, re.IGNORECASE)
    if match:
        opt_num = int(match.group(1))
        if 1 <= opt_num <= len(options):
            answer_letter = chr(64 + opt_num)
    else:
        # Try letter format
        match2 = re.search(r'\b([A-E])\b', output)
        if match2:
            answer_letter = match2.group(1)
    
    return {
        "question": question,
        "options": options,
        "answer": answer_letter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50, help="Number of questions")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "hard_test"])
    parser.add_argument("--output", type=str, default="data/processed/kaggle_eval_input.json")
    args = parser.parse_args()

    split_file = f"data/processed/teleqna_{args.split}.jsonl"
    print(f"Loading {args.split} split from {split_file}...")
    
    with open(split_file) as f:
        raw_items = [json.loads(line) for line in f.readlines()]
    
    items = raw_items[:args.num]
    print(f"Processing {len(items)} questions...")

    # Init retrieval pipeline
    print("Loading retrieval pipeline...")
    retriever = HybridRetriever()
    reranker = Reranker()
    assembler = ContextAssembler()

    eval_data = []
    for i, raw in enumerate(items):
        parsed = parse_teleqna_item(raw)
        
        if not parsed["options"] or not parsed["question"]:
            print(f"  Skipping Q{i+1}: bad parse")
            continue
        
        # Retrieve context with re-ranking
        results = retriever.search(parsed["question"], top_k=10)
        reranked = reranker.rerank(parsed["question"], results, top_k=5, threshold=0.0)
        context = assembler.assemble(reranked)
        
        eval_entry = {
            "question": parsed["question"],
            "options": parsed["options"],
            "answer": parsed["answer"],
            "context": context,
        }
        eval_data.append(eval_entry)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(items)}...")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(eval_data)} questions to {args.output}")
    print(f"Upload this file to Kaggle and use it with 03_rag_generation.ipynb")


if __name__ == "__main__":
    main()
