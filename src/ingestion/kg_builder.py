import networkx as nx
import pickle
from typing import List, Dict, Any
from pathlib import Path

class SectionKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._undirected = None  # cached undirected version

    def build_from_sections(self, all_sections: List[Dict[str, Any]]):
        """Builds a heading-based NetworkX graph from sections."""
        for section in all_sections:
            spec = section.get("spec_number")
            clause_str = section.get("clause_string")
            if not spec or not clause_str:
                continue
                
            node_id = f"{spec}_{clause_str}"
            
            # Add Node
            self.graph.add_node(
                node_id,
                spec_number=spec,
                clause_path=section.get("clause_path"),
                clause_title=section.get("clause_title", ""),
                level=section.get("level", 1)
            )
            
            # Add Parent-Child Edge
            clause_path = section.get("clause_path", [])
            if len(clause_path) > 1:
                parent_clause = ".".join(clause_path[:-1])
                parent_id = f"{spec}_{parent_clause}"
                # Ensure parent exists
                if parent_id in self.graph.nodes:
                    self.graph.add_edge(parent_id, node_id, relationship="parent_of")
            
            # Add Cross-Reference Edges
            cross_refs = section.get("cross_references", [])
            for ref in cross_refs:
                # Basic parsing: "TS 38.331" or "clause 5.1"
                if ref.startswith("TS "):
                    target_spec = ref
                    # Add edge to the root of the target spec
                    target_id = f"{target_spec}_root" # Simplification
                    self.graph.add_edge(node_id, target_id, relationship="cross_references")

    def find_related_sections(self, query_heading: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Finds sections by keyword in title and returns their neighbors."""
        query_heading = query_heading.lower()
        matched_nodes = []
        
        # Simple substring match
        for node, data in self.graph.nodes(data=True):
            if query_heading in data.get("clause_title", "").lower():
                matched_nodes.append(node)
        
        # Limit to prevent explosion on broad keywords
        matched_nodes = matched_nodes[:10]
                
        related = set(matched_nodes)
        
        # Cache undirected graph (built once, reused for all queries)
        if self._undirected is None:
            self._undirected = self.graph.to_undirected()
        
        # Traverse neighbors using cached undirected graph
        for node in matched_nodes:
            try:
                subgraph = nx.ego_graph(self._undirected, node, radius=max_hops)
                related.update(subgraph.nodes())
            except nx.NetworkXError:
                continue
            
        results = []
        for n in related:
            if n in self.graph.nodes:
                results.append({
                    "node_id": n,
                    "data": self.graph.nodes[n]
                })
            
        return results

    def get_cross_references(self, spec_number: str, clause: str) -> List[Dict[str, Any]]:
        node_id = f"{spec_number}_{clause}"
        if node_id not in self.graph:
            return []
            
        out_edges = self.graph.out_edges(node_id, data=True)
        refs = []
        for u, v, data in out_edges:
            if data.get("relationship") == "cross_references":
                refs.append({
                    "target_node": v,
                    "target_data": self.graph.nodes.get(v, {})
                })
        return refs

    def save(self, path: str | Path):
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)

    def load(self, path: str | Path):
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
        self._undirected = None  # invalidate cache

    def get_stats(self) -> Dict[str, int]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges()
        }
