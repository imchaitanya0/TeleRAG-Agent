import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval.fusion import HybridRetriever

def run_tests():
    retriever = HybridRetriever()
    
    test_cases = [
        {
            "query": "What is RRC connection reconfiguration?",
            "expected_spec": "TS 38.331"
        },
        {
            "query": "How does the HARQ process work in the MAC layer?",
            "expected_spec": "TS 38.321"
        },
        {
            "query": "What is the purpose of PDCP?",
            "expected_spec": "TS 38.323"
        }
    ]
    
    print("=== Running Retrieval Validation ===")
    success = 0
    for idx, case in enumerate(test_cases):
        print(f"\nTest {idx+1}: {case['query']}")
        results = retriever.search(case["query"], top_k=5)
        
        # Check if the expected spec is in the top 5 results
        specs_found = [r['payload'].get('spec_number') for r in results[:5]]
        unique_specs = list(dict.fromkeys(specs_found))  # Deduplicate while preserving order
        
        if case["expected_spec"] in specs_found:
            print(f"✅ PASS: Found {case['expected_spec']} in top 5 results.")
            print(f"   All specs in top 5: {unique_specs}")
            success += 1
        else:
            print(f"❌ FAIL: Expected {case['expected_spec']}, but found {unique_specs}")
            
    print(f"\nTotal MRR/Success Rate: {success}/{len(test_cases)} ({(success/len(test_cases))*100:.1f}%)")
    
if __name__ == "__main__":
    run_tests()
