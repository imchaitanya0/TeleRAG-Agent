from typing import List, Dict, Any
import json
from pathlib import Path
import re

class HierarchicalChunker:
    def __init__(self, leaf_min=200, leaf_max=500, section_min=500, section_max=1500):
        self.leaf_min = leaf_min
        self.leaf_max = leaf_max
        self.section_min = section_min
        self.section_max = section_max
        # For simplicity, we use a basic word-count as token approximation.
        # In production, replace with `tiktoken`
    
    def _approx_tokens(self, text: str) -> int:
        return len(text.split())

    def _split_into_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter based on punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]

    def chunk_document(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        section_chunks_map = {} # Maps (spec_number, clause_path_tuple) to section chunk
        
        # Identify parent-child relationships first
        for i, section in enumerate(sections):
            content = section.get("content", "")
            tokens = self._approx_tokens(content)
            
            spec_number = section['spec_number']
            clause_path = tuple(section.get("clause_path", []))
            map_key = (spec_number, clause_path)
            chunk_id = f"{spec_number}_{section['clause_string']}"
            
            # Create a section chunk mapping
            section_chunks_map[map_key] = {
                "chunk_id": f"SEC_{chunk_id}",
                "chunk_tier": "section",
                "spec_number": section["spec_number"],
                "clause_path": section["clause_path"],
                "clause_title": section["clause_title"],
                "content": content,
                "token_count": tokens,
                "child_ids": [],
                "metadata": section # Store raw metadata
            }

        # Process leaves and link to sections
        for section in sections:
            content = section.get("content", "")
            spec_number = section['spec_number']
            clause_path = tuple(section.get("clause_path", []))
            map_key = (spec_number, clause_path)
            sec_chunk = section_chunks_map.get(map_key)
            
            # Simple leaf chunking: if content is large, split it
            sentences = self._split_into_sentences(content)
            
            current_leaf_content = []
            current_leaf_tokens = 0
            leaf_idx = 0
            
            for sentence in sentences:
                sent_tokens = self._approx_tokens(sentence)
                
                if current_leaf_tokens + sent_tokens > self.leaf_max and current_leaf_content:
                    # Save current leaf
                    leaf_text = " ".join(current_leaf_content)
                    leaf_id = f"LEAF_{sec_chunk['chunk_id']}_{leaf_idx}"
                    leaf_chunk = {
                        "chunk_id": leaf_id,
                        "chunk_tier": "leaf",
                        "parent_id": sec_chunk["chunk_id"],
                        "content": leaf_text,
                        "token_count": self._approx_tokens(leaf_text),
                        "metadata": section
                    }
                    all_chunks.append(leaf_chunk)
                    sec_chunk["child_ids"].append(leaf_id)
                    
                    # Reset
                    current_leaf_content = [sentence]
                    current_leaf_tokens = sent_tokens
                    leaf_idx += 1
                else:
                    current_leaf_content.append(sentence)
                    current_leaf_tokens += sent_tokens
            
            # Save final leaf if any
            if current_leaf_content:
                leaf_text = " ".join(current_leaf_content)
                leaf_id = f"LEAF_{sec_chunk['chunk_id']}_{leaf_idx}"
                leaf_chunk = {
                    "chunk_id": leaf_id,
                    "chunk_tier": "leaf",
                    "parent_id": sec_chunk["chunk_id"],
                    "content": leaf_text,
                    "token_count": self._approx_tokens(leaf_text),
                    "metadata": section
                }
                all_chunks.append(leaf_chunk)
                sec_chunk["child_ids"].append(leaf_id)

        # Enforce leaf_min: merge short trailing leaves with previous leaf
        merged_chunks = []
        for chunk in all_chunks:
            if chunk["chunk_tier"] == "leaf" and chunk["token_count"] < self.leaf_min and merged_chunks:
                # Find the previous leaf under the same parent
                prev = None
                for c in reversed(merged_chunks):
                    if c["chunk_tier"] == "leaf" and c.get("parent_id") == chunk.get("parent_id"):
                        prev = c
                        break
                if prev:
                    prev["content"] += " " + chunk["content"]
                    prev["token_count"] = self._approx_tokens(prev["content"])
                    # Remove this chunk's id from parent's child_ids
                    spec = chunk["metadata"].get("spec_number", "")
                    parent_path = tuple(chunk["metadata"].get("clause_path", []))
                    map_key = (spec, parent_path)
                    if map_key in section_chunks_map:
                        cids = section_chunks_map[map_key]["child_ids"]
                        if chunk["chunk_id"] in cids:
                            cids.remove(chunk["chunk_id"])
                    continue
            merged_chunks.append(chunk)
        all_chunks = merged_chunks

        # Add sibling_ids: leaves sharing the same parent are siblings
        parent_to_leaves: dict[str, list[str]] = {}
        for chunk in all_chunks:
            if chunk["chunk_tier"] == "leaf":
                pid = chunk.get("parent_id", "")
                parent_to_leaves.setdefault(pid, []).append(chunk["chunk_id"])

        for chunk in all_chunks:
            if chunk["chunk_tier"] == "leaf":
                pid = chunk.get("parent_id", "")
                siblings = [cid for cid in parent_to_leaves.get(pid, []) if cid != chunk["chunk_id"]]
                chunk["sibling_ids"] = siblings

        # Append all section chunks
        for clause_path, sec_chunk in section_chunks_map.items():
            all_chunks.append(sec_chunk)
            
        return all_chunks

    def generate_summary_chunks(self) -> List[Dict[str, Any]]:
        # TODO: Implement LLM-based summary generation
        return []
