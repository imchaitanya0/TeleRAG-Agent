from pathlib import Path
from typing import List, Dict, Any

from .pdf_parser import TelecomPDFParser, parse_multiple_pdfs
from .html_parser import TelecomHTMLParser
from .text_parser import TelecomTextParser

def parse_document(file_path: str | Path) -> List[Dict[str, Any]]:
    """
    Router function to detect file type and apply the appropriate parser.
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.pdf':
        parser = TelecomPDFParser()
        return parser.parse(file_path)
    elif file_path.suffix.lower() in ['.html', '.htm']:
        parser = TelecomHTMLParser()
        return parser.parse(file_path)
    elif file_path.suffix.lower() == '.txt':
        parser = TelecomTextParser()
        return parser.parse(file_path)
    else:
        print(f"Unsupported file format: {file_path.suffix}")
        return []
