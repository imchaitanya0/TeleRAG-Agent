"""
Text Parser for 3GPP specifications stored as structured plain text files.
Reuses the same section-detection logic as the PDF parser.
"""
import re
from typing import List, Dict, Any
from pathlib import Path


class TelecomTextParser:
    def __init__(self):
        self.section_regex = re.compile(r"^(\d+\.)+\s+(.+)$")
        self.ts_ref_regex = re.compile(r"TS\s+(\d{2}\.\d{3})")
        self.clause_ref_regex = re.compile(r"clause\s+(\d+(\.\d+)*)")

    def parse(self, text_path: str | Path) -> List[Dict[str, Any]]:
        """Parses a plain-text 3GPP specification into structured sections."""
        text_path = Path(text_path)
        sections = []

        try:
            with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"Error opening {text_path}: {e}")
            return []

        # Extract spec number from filename (e.g., TS_38.331.txt)
        spec_number = "Unknown"
        match = re.search(r"TS_(\d{2}\.\d{3})", text_path.name)
        if match:
            spec_number = f"TS {match.group(1)}"

        current_section = None
        current_content = []

        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                if current_section:
                    current_content.append("")
                continue

            header_match = self.section_regex.match(line_stripped)
            if header_match:
                # Save previous section
                if current_section:
                    current_section["content"] = "\n".join(current_content)
                    sections.append(current_section)

                clause_str = header_match.group(1).strip(".")
                clause_path = clause_str.split(".")
                clause_title = header_match.group(2).strip()

                current_section = {
                    "spec_number": spec_number,
                    "release": "16",
                    "clause_path": clause_path,
                    "clause_string": clause_str,
                    "clause_title": clause_title,
                    "level": len(clause_path),
                    "tables": [],
                    "cross_references": [],
                }
                current_content = [line_stripped]
            else:
                if current_section:
                    current_content.append(line_stripped)
                    # Cross-reference detection
                    ts_refs = self.ts_ref_regex.findall(line_stripped)
                    clause_refs = self.clause_ref_regex.findall(line_stripped)
                    if ts_refs:
                        current_section["cross_references"].extend(
                            [f"TS {ref}" for ref in ts_refs]
                        )
                    if clause_refs:
                        current_section["cross_references"].extend(
                            [f"clause {ref[0]}" for ref in clause_refs]
                        )

        # Append last section
        if current_section:
            current_section["content"] = "\n".join(current_content)
            current_section["cross_references"] = list(
                set(current_section["cross_references"])
            )
            sections.append(current_section)

        return sections
