import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_assembler import ContextAssembler

def main():
    print("Loading RAG Pipeline...")
    retriever = HybridRetriever()
    reranker = Reranker()
    assembler = ContextAssembler()
    
    print("Loading golden questions...")
    with open("data/processed/kaggle_eval_golden_50.json", "r") as f:
        golden_qs = json.load(f)
        
    eval_data = []
    
    print("Retrieving context for 50 golden questions...")
    for i, item in enumerate(tqdm(golden_qs)):
        q = item["question"]
        
        # 1. Retrieve
        results = retriever.search(q, top_k=10)
        
        # 2. Rerank
        reranked = reranker.rerank(q, results, top_k=5, threshold=0.0)
        
        # 3. Assemble
        context = assembler.assemble(reranked)
        
        eval_data.append({
            "question": q,
            "options": item["options"],
            "answer": item["answer"],
            "context": context
        })
        
    with open("data/processed/kaggle_eval_golden_final.json", "w") as f:
        json.dump(eval_data, f, indent=2)
        
    print("\nSaved fully processed golden dataset to: data/processed/kaggle_eval_golden_final.json")
    print("Upload this file to Kaggle to test the RAG pipeline.")

if __name__ == "__main__":
    main()
