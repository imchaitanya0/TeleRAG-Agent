import sys
from pathlib import Path

# Add root to python path if run directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.kg_builder import SectionKnowledgeGraph

class KGSearcher:
    def __init__(self, graph_path: str = "data/processed/section_graph.pkl"):
        self.kg = SectionKnowledgeGraph()
        path = Path(graph_path)
        if path.exists():
            self.kg.load(path)
        else:
            print(f"Warning: KG file {graph_path} not found.")

    def search(self, query: str, max_hops: int = 2) -> list:
        """Finds related sections in the KG based on keyword matching of title, node_id, and content keywords."""
        query_lower = query.lower()
        # Remove common stop words for naive matching
        stop_words = {"what", "is", "the", "how", "to", "did", "why", "a", "an", "and", "or", "in", "of", "for"}
        words = [w for w in query_lower.split() if w not in stop_words and len(w) > 1]
        
        # Map common concepts to spec numbers to find entry nodes
        spec_map = {
            "rrc": "38.331",
            "mac": "38.321",
            "harq": "38.321",
            "drx": "38.321",
            "rlc": "38.322",
            "pdcp": "38.323",
            "sdap": "37.324",
            "phy": "38.211",
            "physical": "38.211",
            "ng-ran": "38.300",
            "architecture": "38.300",
            "handover": "38.133",
            "rrm": "38.133",
            "power": "38.213",
            "control": "38.213"
        }
        
        # Inject spec numbers into the search words if acronyms match
        search_terms = list(words)
        for w in words:
            if w in spec_map:
                search_terms.append(spec_map[w])
        
        results = []
        for word in search_terms:
            # Match against title, node_id, and spec-related content
            for node, data in self.kg.graph.nodes(data=True):
                node_lower = node.lower()
                title_lower = data.get("clause_title", "").lower()
                if word in title_lower or word in node_lower:
                    results.append({"node_id": node, "data": data})
            
            # Also use the built-in method for broader traversal
            related = self.kg.find_related_sections(word, max_hops=max_hops)
            results.extend(related)
            
        # Deduplicate
        seen = set()
        unique_results = []
        for res in results:
            node_id = res["node_id"]
            if node_id not in seen:
                seen.add(node_id)
                unique_results.append(res)
                
        return unique_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search Knowledge Graph for section nodes matching a keyword")
    parser.add_argument("query", type=str, help="Keyword to search (e.g., RRC, HARQ, PDCP, MAC)")
    parser.add_argument("--max_hops", type=int, default=2, help="Maximum hops for traversal (default 2)")
    args = parser.parse_args()
    
    searcher = KGSearcher()
    results = searcher.search(args.query, max_hops=args.max_hops)
    
    print(f"\n=== KG Search Results for '{args.query}' ===")
    print(f"Found {len(results)} related section nodes")
    for r in results[:5]:
        print(f"Node: {r['node_id']} | Title: {r['data'].get('clause_title')}")