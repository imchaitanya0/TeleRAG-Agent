import pytest
from src.ingestion.chunker import HierarchicalChunker

def test_hierarchical_chunker():
    # Setup dummy enriched sections
    sections = [
        {
            "spec_number": "TS 38.331",
            "clause_path": ["5", "3", "3"],
            "clause_string": "5.3.3",
            "clause_title": "RRC connection reconfiguration",
            "content": "This is a short sentence. " * 30, # ~120 words
        },
        {
            "spec_number": "TS 38.331",
            "clause_path": ["5", "3", "3", "1"],
            "clause_string": "5.3.3.1",
            "clause_title": "General",
            "content": "A very long text that should be split into multiple leaves. " * 150 # ~1200 words
        }
    ]
    
    chunker = HierarchicalChunker(leaf_max=100) # Lowering max for testing purposes
    chunks = chunker.chunk_document(sections)
    
    # Separate leaves and sections
    leaves = [c for c in chunks if c["chunk_tier"] == "leaf"]
    section_chunks = [c for c in chunks if c["chunk_tier"] == "section"]
    
    # Assertions
    assert len(section_chunks) == 2, "Should create a section chunk for each clause"
    assert len(leaves) > 2, "Long content should be split into multiple leaves"
    
    # Check linking
    for leaf in leaves:
        assert "parent_id" in leaf
        assert leaf["parent_id"].startswith("SEC_TS 38.331_5.3.3")
        
    for sec in section_chunks:
        assert "child_ids" in sec
        assert isinstance(sec["child_ids"], list)
