import json
from pathlib import Path
from typing import List, Dict

class ContextAssembler:
    def __init__(self, chunks_file: str = "data/processed/chunks.jsonl"):
        self.chunk_db = {}
        self._load_chunks(chunks_file)

    def _load_chunks(self, chunks_file: str):
        path = Path(chunks_file)
        if not path.exists():
            print(f"Warning: Chunks file not found at {chunks_file}")
            return
            
        with open(path, "r") as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    self.chunk_db[chunk["chunk_id"]] = chunk
                except json.JSONDecodeError:
                    continue

    def assemble(self, ranked_candidates: List[Dict], max_tokens: int = 3500) -> str:
        """
        Assembles expanded context from ranked candidates, applying parent-child expansion.
        Enforces a max_tokens budget.
        """
        assembled_chunks = []
        seen_ids = set()
        current_tokens = 0
        
        for cand in ranked_candidates:
            if current_tokens >= max_tokens:
                break
                
            chunk_id = cand["chunk_id"]
            if chunk_id in seen_ids:
                continue
                
            # If we don't have it in our DB (shouldn't happen), use the payload directly
            if chunk_id not in self.chunk_db:
                chunk_data = cand["payload"]
                token_count = chunk_data.get("token_count", 0)
                if current_tokens + token_count <= max_tokens:
                    assembled_chunks.append(chunk_data)
                    seen_ids.add(chunk_id)
                    current_tokens += token_count
                continue

            chunk_data = self.chunk_db[chunk_id]
            
            # Expansion logic: If it's a leaf, try to include the parent section and siblings
            # to provide full context, up to the token limit.
            if chunk_data.get("chunk_tier") == "leaf":
                parent_id = chunk_data.get("parent_id")
                
                # If there's a parent, grab the parent and all its leaf children (siblings)
                # to reconstruct the full section, if it fits.
                if parent_id and parent_id in self.chunk_db and parent_id not in seen_ids:
                    parent_chunk = self.chunk_db[parent_id]
                    family_tokens = parent_chunk.get("token_count", 0)
                    
                    # Calculate total tokens for the entire section (parent + all leaves)
                    family_members = [parent_chunk]
                    for child_id in parent_chunk.get("child_ids", []):
                        if child_id in self.chunk_db:
                            child = self.chunk_db[child_id]
                            family_tokens += child.get("token_count", 0)
                            family_members.append(child)
                            
                    # If the whole section fits, add it all
                    if current_tokens + family_tokens <= max_tokens:
                        for member in family_members:
                            if member["chunk_id"] not in seen_ids:
                                assembled_chunks.append(member)
                                seen_ids.add(member["chunk_id"])
                        current_tokens += family_tokens
                        continue
                        
            # If it's a section, or if family expansion didn't fit, just add the chunk itself
            token_count = chunk_data.get("token_count", 0)
            if current_tokens + token_count <= max_tokens:
                assembled_chunks.append(chunk_data)
                seen_ids.add(chunk_id)
                current_tokens += token_count

        return self._format_context(assembled_chunks)

    def _format_context(self, chunks: List[Dict]) -> str:
        """Formats the selected chunks into a clean, readable string with citations."""
        formatted_pieces = []
        for c in chunks:
            spec = c.get("spec_number", "Unknown Spec")
            clause = c.get("clause_string", "")
            title = c.get("clause_title")
            content = c.get("content", "").strip()
            
            header = f"[Source: {spec}, §{clause}"
            if title and title != "None":
                header += f" - {title}"
            header += "]"
            
            formatted_pieces.append(f"{header}\n{content}")
            
        return "\n\n".join(formatted_pieces)

if __name__ == "__main__":
    # Test assembler
    assembler = ContextAssembler()
    
    # Mock some ranked candidates
    mock_candidates = [
        {"chunk_id": "LEAF_SEC_TS 38.331_5_0", "payload": {}}
    ]
    
    context = assembler.assemble(mock_candidates)
    print("=== Assembled Context ===")
    print(context)
    print(f"\nTotal length: {len(context)} characters")
