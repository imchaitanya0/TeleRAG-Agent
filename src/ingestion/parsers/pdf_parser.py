"""
Structure-aware 3GPP PDF parser.

Uses PyMuPDF's built-in TOC (Table of Contents / bookmarks) for reliable
section detection instead of fragile regex-based heading matching.
Each section's text is extracted by slicing between TOC page boundaries.
"""
import re
import fitz  # PyMuPDF
import concurrent.futures
from typing import List, Dict, Any
from pathlib import Path


class TelecomPDFParser:
    """Parses 3GPP/ETSI specification PDFs into structured sections."""

    def __init__(self, skip_boilerplate: bool = True):
        """
        Args:
            skip_boilerplate: If True, skip front-matter sections like
                Intellectual Property Rights, Legal Notice, Foreword, etc.
        """
        self.skip_boilerplate = skip_boilerplate
        self.ts_ref_regex = re.compile(r"TS\s+(\d{2}\.\d{3})")
        self.clause_ref_regex = re.compile(r"clause\s+(\d+(?:\.\d+)*)")
        # Section titles to skip (front-matter, not spec content)
        self._skip_titles = {
            "intellectual property rights", "legal notice",
            "modal verbs terminology", "foreword", "contents",
        }

    def _extract_spec_number(self, pdf_path: Path) -> str:
        """Extract spec number from filename like ts_138331v180501p.pdf → TS 38.331"""
        name = pdf_path.stem.lower()
        # Pattern: ts_1XXYYYV... → TS XX.YYY
        match = re.search(r"ts_1(\d{2})(\d{3})", name)
        if match:
            return f"TS {match.group(1)}.{match.group(2)}"
        # Fallback pattern: TS_38.331
        match2 = re.search(r"TS_(\d{2}\.\d{3})", pdf_path.name, re.IGNORECASE)
        if match2:
            return f"TS {match2.group(1)}"
        return "Unknown"

    def _extract_release(self, pdf_path: Path) -> str:
        """Extract release number from filename or first page."""
        name = pdf_path.stem.lower()
        # ts_138331v180501p → v18 → Release 18
        match = re.search(r"v(\d{2})", name)
        if match:
            return match.group(1)
        return "18"  # default

    def _should_skip_section(self, title: str) -> bool:
        """Returns True if this section is front-matter boilerplate."""
        if not self.skip_boilerplate:
            return False
        return title.strip().lower() in self._skip_titles

    def _extract_clause_string(self, title: str) -> tuple[str, str]:
        """
        Parse '5.3.3 RRC connection reconfiguration' → ('5.3.3', 'RRC connection reconfiguration')
        Returns (clause_string, clean_title). If no clause number, returns ('', title).
        """
        match = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", title.strip())
        if match:
            return match.group(1), match.group(2).strip()
        return "", title.strip()

    def parse(self, pdf_path: str | Path) -> List[Dict[str, Any]]:
        """
        Parse a 3GPP PDF into structured sections using the PDF's TOC.

        Returns a list of section dicts, each with:
            - spec_number, release, clause_path, clause_string, clause_title
            - level (depth in hierarchy)
            - content (full text of the section)
            - cross_references (list of referenced specs/clauses)
        """
        pdf_path = Path(pdf_path)
        sections = []

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  ERROR opening {pdf_path.name}: {e}")
            return []

        spec_number = self._extract_spec_number(pdf_path)
        release = self._extract_release(pdf_path)
        toc = doc.get_toc()  # [(level, title, page_number), ...]
        total_pages = len(doc)

        if not toc:
            print(f"  WARNING: {pdf_path.name} has no TOC, falling back to regex-based parsing")
            sections = self._parse_without_toc(doc, pdf_path, spec_number, release)
            doc.close()
            return sections

        # Build section list from TOC with page ranges
        toc_entries = []
        for i, (level, title, start_page) in enumerate(toc):
            if self._should_skip_section(title):
                continue
            # End page = start of next section (or end of document)
            end_page = total_pages
            for j in range(i + 1, len(toc)):
                end_page = toc[j][2]
                break
            toc_entries.append({
                "level": level,
                "title": title,
                "start_page": start_page,
                "end_page": end_page,
            })

        # Extract text for each section
        for entry in toc_entries:
            title = entry["title"]
            clause_str, clean_title = self._extract_clause_string(title)
            clause_path = clause_str.split(".") if clause_str else []

            # Extract text from the page range (0-indexed internally)
            start_pg = max(0, entry["start_page"] - 1)
            end_pg = min(total_pages, entry["end_page"] - 1)

            text_parts = []
            for pg in range(start_pg, end_pg + 1):
                page_text = doc[pg].get_text("text")
                # Strip ETSI header/footer lines that appear on every page
                lines = page_text.split("\n")
                cleaned_lines = []
                for line in lines:
                    stripped = line.strip()
                    # Skip ETSI headers, page numbers, spec title repeats
                    if stripped.startswith("ETSI TS "):
                        continue
                    if stripped.startswith("ETSI"):
                        continue
                    if stripped.startswith("3GPP TS "):
                        continue
                    if re.match(r"^\d+$", stripped):  # bare page number
                        continue
                    cleaned_lines.append(line)
                text_parts.append("\n".join(cleaned_lines))

            content = "\n".join(text_parts).strip()

            # Skip near-empty sections
            if len(content) < 20:
                continue

            # Extract cross-references
            cross_refs = list(set(
                [f"TS {ref}" for ref in self.ts_ref_regex.findall(content)] +
                [f"clause {ref[0]}" for ref in self.clause_ref_regex.findall(content)]
            ))

            section = {
                "spec_number": spec_number,
                "release": release,
                "clause_path": clause_path,
                "clause_string": clause_str,
                "clause_title": clean_title,
                "level": entry["level"],
                "content": content,
                "tables": [],
                "cross_references": cross_refs,
            }
            sections.append(section)

        doc.close()
        return sections

    def _parse_without_toc(self, doc, pdf_path, spec_number, release):
        """Fallback: regex-based parsing for PDFs without TOC bookmarks."""
        heading_re = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
        sections = []
        current_section = None
        current_content = []

        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            for line in text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("ETSI") or stripped.startswith("3GPP TS"):
                    continue
                if re.match(r"^\d+$", stripped):
                    continue

                m = heading_re.match(stripped)
                if m and len(stripped) < 120:
                    if current_section:
                        current_section["content"] = "\n".join(current_content)
                        if len(current_section["content"]) >= 20:
                            sections.append(current_section)

                    clause_str = m.group(1)
                    clause_title = m.group(2).strip()

                    current_section = {
                        "spec_number": spec_number,
                        "release": release,
                        "clause_path": clause_str.split("."),
                        "clause_string": clause_str,
                        "clause_title": clause_title,
                        "level": clause_str.count(".") + 1,
                        "content": "",
                        "tables": [],
                        "cross_references": [],
                    }
                    current_content = [stripped]
                elif current_section:
                    current_content.append(stripped)
                    ts_refs = self.ts_ref_regex.findall(stripped)
                    clause_refs = self.clause_ref_regex.findall(stripped)
                    if ts_refs:
                        current_section["cross_references"].extend([f"TS {r}" for r in ts_refs])
                    if clause_refs:
                        current_section["cross_references"].extend([f"clause {r[0]}" for r in clause_refs])

        if current_section:
            current_section["content"] = "\n".join(current_content)
            current_section["cross_references"] = list(set(current_section["cross_references"]))
            if len(current_section["content"]) >= 20:
                sections.append(current_section)

        return sections


def parse_multiple_pdfs(pdf_paths: List[Path], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    ⚡ Parallel parsing of PDFs using ThreadPoolExecutor.
    """
    parser = TelecomPDFParser()
    all_sections = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(parser.parse, pdf_paths)
        for idx, doc_sections in enumerate(results):
            print(f"  Parsed {pdf_paths[idx].name}: {len(doc_sections)} sections")
            all_sections.extend(doc_sections)

    return all_sections


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = TelecomPDFParser()
        secs = parser.parse(sys.argv[1])
        print(f"Extracted {len(secs)} sections.")
        for s in secs[:5]:
            print(f"  [{s['clause_string']}] {s['clause_title']} ({len(s['content'])} chars)")
