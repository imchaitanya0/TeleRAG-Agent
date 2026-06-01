import pytest
import os
import tempfile
from src.ingestion.kg_builder import SectionKnowledgeGraph

@pytest.fixture
def sample_sections():
    return [
        {
            "spec_number": "TS 38.331",
            "clause_path": ["5"],
            "clause_string": "5",
            "clause_title": "Procedures",
            "cross_references": []
        },
        {
            "spec_number": "TS 38.331",
            "clause_path": ["5", "3"],
            "clause_string": "5.3",
            "clause_title": "Connection control",
            "cross_references": []
        },
        {
            "spec_number": "TS 38.331",
            "clause_path": ["5", "3", "3"],
            "clause_string": "5.3.3",
            "clause_title": "RRC connection reconfiguration",
            "cross_references": ["TS 38.321", "clause 5.1"]
        }
    ]

def test_kg_build_and_query(sample_sections):
    kg = SectionKnowledgeGraph()
    kg.build_from_sections(sample_sections)
    
    stats = kg.get_stats()
    assert stats["nodes"] == 4 # 3 sections + 1 implicit cross-ref root
    assert stats["edges"] > 0
    
    # Test related sections
    results = kg.find_related_sections("RRC")
    assert len(results) > 0
    
    found_node = [r for r in results if r["node_id"] == "TS 38.331_5.3.3"]
    assert len(found_node) == 1
    
    # Test cross references
    refs = kg.get_cross_references("TS 38.331", "5.3.3")
    assert len(refs) > 0

def test_kg_serialization(sample_sections):
    kg = SectionKnowledgeGraph()
    kg.build_from_sections(sample_sections)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        
    kg.save(tmp_path)
    
    kg2 = SectionKnowledgeGraph()
    kg2.load(tmp_path)
    
    assert kg.get_stats() == kg2.get_stats()
    
    os.unlink(tmp_path)
