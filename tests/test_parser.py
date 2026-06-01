import pytest
import tempfile
import json
from pathlib import Path
from src.ingestion.parsers.pdf_parser import TelecomPDFParser
from src.ingestion.parsers.html_parser import TelecomHTMLParser
from src.ingestion.metadata_enricher import enrich_section
from reportlab.pdfgen import canvas

@pytest.fixture
def dummy_pdf():
    # Create a temporary PDF file with dummy 3GPP content using reportlab
    # This is a robust way to test PyMuPDF extraction
    fd, path = tempfile.mkstemp(suffix=".pdf")
    
    c = canvas.Canvas(path)
    c.drawString(100, 800, "TS 38.331")
    c.drawString(100, 780, "5.3.3 RRC connection reconfiguration")
    c.drawString(100, 760, "The UE shall apply the configuration. See TS 38.321 clause 5.1.")
    c.drawString(100, 740, "5.3.3.1 General")
    c.drawString(100, 720, "General information about RRC.")
    c.save()
    
    yield path
    
    Path(path).unlink(missing_ok=True)

@pytest.fixture
def dummy_html():
    fd, path = tempfile.mkstemp(suffix=".html")
    html_content = """
    <html>
        <body>
            <h1>5.3.3 RRC connection reconfiguration</h1>
            <p>The UE shall apply the configuration.</p>
        </body>
    </html>
    """
    with open(path, "w") as f:
        f.write(html_content)
        
    yield path
    Path(path).unlink(missing_ok=True)

def test_pdf_parser(dummy_pdf):
    parser = TelecomPDFParser()
    sections = parser.parse(dummy_pdf)
    
    assert len(sections) == 2
    assert sections[0]["clause_string"] == "5.3.3"
    assert sections[0]["clause_title"] == "RRC connection reconfiguration"
    assert "TS 38.321" in sections[0]["cross_references"]
    
    assert sections[1]["clause_string"] == "5.3.3.1"
    assert sections[1]["clause_title"] == "General"

def test_html_parser(dummy_html):
    parser = TelecomHTMLParser()
    sections = parser.parse(dummy_html)
    
    assert len(sections) == 1
    assert sections[0]["clause_string"] == "5.3.3"
    assert sections[0]["clause_title"] == "RRC connection reconfiguration"

def test_metadata_enricher():
    raw_section = {
        "spec_number": "TS 38.331",
        "clause_title": "RRC connection reconfiguration",
        "content": "This is a test content with multiple words."
    }
    
    enriched = enrich_section(raw_section)
    
    assert enriched["tsg"] == "RAN"
    assert enriched["working_group"] == "RAN2"
    assert enriched["content_type"] == "normative"
    assert enriched["token_count"] == 8
    assert "chunk_tier" in enriched
