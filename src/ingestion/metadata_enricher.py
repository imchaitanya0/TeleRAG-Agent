from typing import Dict, Any

# Mapping of specs to Technical Specification Groups (TSG) and Working Groups (WG)
SPEC_TO_TSG = {
    "38.300": "RAN", "38.331": "RAN", "38.321": "RAN", "38.322": "RAN",
    "38.323": "RAN", "38.211": "RAN", "38.213": "RAN", "38.133": "RAN"
}

SPEC_TO_WG = {
    "38.300": "RAN2", "38.331": "RAN2", "38.321": "RAN2", "38.322": "RAN2",
    "38.323": "RAN2", "38.211": "RAN1", "38.213": "RAN1", "38.133": "RAN4"
}

def determine_content_type(title: str) -> str:
    title_lower = title.lower()
    if any(kw in title_lower for kw in ["definition", "abbreviation", "acronym"]):
        return "definition"
    if any(kw in title_lower for kw in ["overview", "introduction", "general"]):
        return "informative"
    return "normative"

def enrich_section(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches a parsed section dictionary to ensure it has the required 12 metadata fields.
    """
    spec_num_str = section.get("spec_number", "")
    
    # Extract just the number, e.g., "38.331" from "TS 38.331"
    raw_num = spec_num_str.replace("TS ", "").strip()
    
    tsg = SPEC_TO_TSG.get(raw_num, "Unknown")
    wg = SPEC_TO_WG.get(raw_num, "Unknown")
    
    content_type = determine_content_type(section.get("clause_title", ""))
    
    # 1. spec_number (already present)
    # 2. release (already present)
    # 3. clause_path (already present)
    # 4. clause_title (already present)
    section["tsg"] = tsg # 5
    section["working_group"] = wg # 6
    section["content_type"] = content_type # 7
    # 8. cross_references (already present)
    section["entities"] = [] # 9. Placeholder for now
    section["chunk_tier"] = "unassigned" # 10. To be set by chunker
    section["parent_id"] = None # 11. To be set by chunker
    
    # 12. Token count approximation
    content = section.get("content", "")
    section["token_count"] = len(content.split())
    
    return section
