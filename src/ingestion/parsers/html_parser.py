import re
from typing import List, Dict, Any
from pathlib import Path
from bs4 import BeautifulSoup

class TelecomHTMLParser:
    def __init__(self):
        # Regex to detect 3GPP section headers in text if tags aren't sufficient
        self.section_regex = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")

    def parse(self, html_path: str | Path) -> List[Dict[str, Any]]:
        """Parses a 3GPP HTML specification into structured sections."""
        html_path = Path(html_path)
        sections = []
        
        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")
        except Exception as e:
            print(f"Error opening {html_path}: {e}")
            return []

        # Basic metadata
        spec_number = "Unknown"
        match = re.search(r"TS_(\d{2}\.\d{3})", html_path.name)
        if match:
            spec_number = f"TS {match.group(1)}"

        # Find all headers (h1-h6) to denote sections
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for i, header in enumerate(headers):
            header_text = header.get_text(strip=True)
            match = self.section_regex.match(header_text)
            
            if match:
                clause_str = match.group(1).strip('.')
                clause_path = clause_str.split('.')
                clause_title = match.group(2).strip()
            else:
                # Fallback if header doesn't match standard numbering
                clause_str = f"H{i}"
                clause_path = [clause_str]
                clause_title = header_text

            # Extract content between this header and the next
            content_elements = []
            current_sibling = header.find_next_sibling()
            
            while current_sibling and current_sibling.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                content_elements.append(current_sibling.get_text(strip=True))
                current_sibling = current_sibling.find_next_sibling()
                
            content = "\n".join([c for c in content_elements if c])
            
            # Very basic cross-reference extraction
            cross_references = []
            ts_refs = re.findall(r"TS\s+(\d{2}\.\d{3})", content)
            clause_refs = re.findall(r"clause\s+(\d+(\.\d+)*)", content)
            
            if ts_refs:
                cross_references.extend([f"TS {ref}" for ref in ts_refs])
            if clause_refs:
                cross_references.extend([f"clause {ref[0]}" for ref in clause_refs])

            section = {
                "spec_number": spec_number,
                "release": "16", # Default
                "clause_path": clause_path,
                "clause_string": clause_str,
                "clause_title": clause_title,
                "level": len(clause_path),
                "content": content,
                "tables": [], # Could extract HTML tables here
                "cross_references": list(set(cross_references))
            }
            sections.append(section)
            
        return sections
