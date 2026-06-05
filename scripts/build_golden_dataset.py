import sys
import json
import re
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval.dense_search import DenseSearcher
from src.ingestion.embedder import TelecomEmbedder

def build_golden_dataset():
    print("Loading TeleQnA...")
    with open("data/raw/teleqna/TeleQnA.json", "r") as f:
        raw_data = json.load(f)
        
    print("Initializing DenseSearcher and Embedder...")
    searcher = DenseSearcher()
    embedder = TelecomEmbedder()
    
    scored_questions = []
    
    # Only check Standards specifications questions
    standards_qs = [
        (k, v) for k, v in raw_data.items() 
        if v.get("category") == "Standards specifications"
    ]
    
    print(f"Testing retrieval confidence for {len(standards_qs)} questions...")
    
    # We'll test the first 500 to find the best 50
    for k, v in tqdm(standards_qs[:500]):
        question_text = v["question"]
        
        # Embed question
        dense_vec, _ = embedder.embed_query(question_text)
        
        # Search DB
        results = searcher.search(dense_vec, limit=1)
        
        if results:
            best_score = results[0]["score"]
            scored_questions.append({
                "id": k,
                "score": best_score,
                "data": v
            })
            
    # Sort by highest similarity score
    scored_questions.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top 50
    golden_50 = scored_questions[:50]
    
    print(f"\nFound top 50 golden questions!")
    print(f"Highest score: {golden_50[0]['score']:.4f}")
    print(f"Lowest score in top 50: {golden_50[-1]['score']:.4f}")
    
    # Format exactly like Kaggle eval dataset
    output_data = []
    for item in golden_50:
        v = item["data"]
        
        # Parse answer letter (A, B, C...)
        ans_str = v["answer"]
        match = re.search(r"option\s+(\d+)", ans_str, re.IGNORECASE)
        opt_idx = int(match.group(1)) - 1 if match else 0
        
        options = [v[f"option {i}"] for i in range(1, 10) if f"option {i}" in v]
        
        output_data.append({
            "question": v["question"],
            "options": options,
            "answer": chr(65 + opt_idx),
            "explanation": v.get("explanation", ""),
            "retrieval_confidence": item["score"]
        })
        
    with open("data/processed/kaggle_eval_golden_50.json", "w") as f:
        json.dump(output_data, f, indent=2)
        
    print("\nSaved to: data/processed/kaggle_eval_golden_50.json")

if __name__ == "__main__":
    build_golden_dataset()
