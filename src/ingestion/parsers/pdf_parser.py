import re
import fitz  # PyMuPDF
import pdfplumber
import concurrent.futures
from typing import List, Dict, Any
from pathlib import Path

class TelecomPDFParser:
    def __init__(self):
        # Regex to detect 3GPP section headers like "5.3.3 RRC connection reconfiguration"
        # Matches patterns like "1. ", "5.3 ", "10.2.1.4 " at the start of a line.
        self.section_regex = re.compile(r"^(\d+\.)+\s+(.+)$")
        # Regex to detect cross-references
        self.ts_ref_regex = re.compile(r"TS\s+(\d{2}\.\d{3})")
        self.clause_ref_regex = re.compile(r"clause\s+(\d+(\.\d+)*)")

    def parse(self, pdf_path: str | Path) -> List[Dict[str, Any]]:
        """Parses a single PDF document into structured sections."""
        pdf_path = Path(pdf_path)
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening {pdf_path}: {e}")
            return []

        current_section = None
        current_content = []

        # Extract basic metadata from filename (e.g., TS_38.331.pdf)
        spec_number = "Unknown"
        match = re.search(r"TS_(\d{2}\.\d{3})", pdf_path.name)
        if match:
            spec_number = f"TS {match.group(1)}"

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line is a section header
                header_match = self.section_regex.match(line)
                if header_match:
                    # Save previous section
                    if current_section:
                        current_section["content"] = "\n".join(current_content)
                        sections.append(current_section)
                    
                    # Start new section
                    clause_str = header_match.group(1).strip('.')
                    clause_path = clause_str.split('.')
                    clause_title = header_match.group(2).strip()
                    
                    current_section = {
                        "spec_number": spec_number,
                        "release": "16", # Defaulting or extract if possible
                        "clause_path": clause_path,
                        "clause_string": clause_str,
                        "clause_title": clause_title,
                        "level": len(clause_path),
                        "tables": [], # Would extract with pdfplumber here
                        "cross_references": []
                    }
                    current_content = [line]
                else:
                    if current_section:
                        current_content.append(line)
                        # Look for cross-references
                        ts_refs = self.ts_ref_regex.findall(line)
                        clause_refs = self.clause_ref_regex.findall(line)
                        if ts_refs:
                            current_section["cross_references"].extend([f"TS {ref}" for ref in ts_refs])
                        if clause_refs:
                            current_section["cross_references"].extend([f"clause {ref[0]}" for ref in clause_refs])

        # Append the last section
        if current_section:
            current_section["content"] = "\n".join(current_content)
            # Deduplicate cross references
            current_section["cross_references"] = list(set(current_section["cross_references"]))
            sections.append(current_section)
            
        doc.close()
        
        # Extract tables using pdfplumber and attach to nearest section
        try:
            with pdfplumber.open(pdf_path) as pdf_pl:
                for page in pdf_pl.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if not table:
                            continue
                        # Convert table to markdown
                        md_rows = []
                        for row_idx, row in enumerate(table):
                            cleaned = [str(cell).strip() if cell else "" for cell in row]
                            md_rows.append("| " + " | ".join(cleaned) + " |")
                            if row_idx == 0:
                                md_rows.append("| " + " | ".join(["---"] * len(cleaned)) + " |")
                        md_table = "\n".join(md_rows)
                        
                        # Attach to the last section (best approximation)
                        if sections:
                            sections[-1]["tables"].append(md_table)
        except Exception as e:
            # pdfplumber may fail on some PDFs — not critical
            print(f"Table extraction warning for {pdf_path.name}: {e}")
        
        return sections

def parse_multiple_pdfs(pdf_paths: List[Path], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    ⚡ BOTTLENECK #2 MITIGATION: Parallel parsing of PDFs
    Reduces ingestion time from hours to minutes.
    """
    parser = TelecomPDFParser()
    all_sections = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map returns an iterator of results, keeping order
        results = executor.map(parser.parse, pdf_paths)
        for idx, doc_sections in enumerate(results):
            print(f"Parsed {pdf_paths[idx].name}: {len(doc_sections)} sections")
            all_sections.extend(doc_sections)
            
    return all_sections

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        parser = TelecomPDFParser()
        secs = parser.parse(sys.argv[1])
        print(f"Extracted {len(secs)} sections.")
