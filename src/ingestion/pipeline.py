import os
import json
from pathlib import Path
from tqdm import tqdm

from src.ingestion.parsers import parse_document
from src.ingestion.metadata_enricher import enrich_section
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.kg_builder import SectionKnowledgeGraph

def run_ingestion(raw_data_dir: Path, output_dir: Path):
    print("=== Starting End-to-End Ingestion ===")
    
    # Ensure dirs exist
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_3gpp_dir = raw_data_dir / "3gpp"
    
    if not raw_3gpp_dir.exists():
        print(f"Directory {raw_3gpp_dir} not found. Ensure data is downloaded.")
        return

    # Find all PDFs and HTMLs
    files = list(raw_3gpp_dir.glob("*.pdf")) + list(raw_3gpp_dir.glob("*.html"))
    if not files:
        print("No documents found to parse.")
        return

    print(f"Found {len(files)} documents.")
    
    all_sections = []
    
    # Step 2: Parse each file
    print("Parsing documents...")
    for file_path in tqdm(files):
        # We process sequentially here for the stub. 
        # In a real run, `parse_multiple_pdfs` from pdf_parser could be invoked instead.
        sections = parse_document(file_path)
        all_sections.extend(sections)

    # Step 3: Enrich metadata
    print("Enriching metadata...")
    enriched_sections = []
    for section in tqdm(all_sections):
        enriched = enrich_section(section)
        enriched_sections.append(enriched)

    # Step 4: Run hierarchical chunking
    print("Running hierarchical chunking...")
    chunker = HierarchicalChunker()
    chunks = chunker.chunk_document(enriched_sections)

    # Step 5: Build KG
    print("Building Section Knowledge Graph...")
    kg = SectionKnowledgeGraph()
    kg.build_from_sections(enriched_sections)

    # Step 6: Save outputs
    print("Saving outputs...")
    chunks_file = output_dir / "chunks.jsonl"
    with open(chunks_file, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
            
    kg_file = output_dir / "section_graph.pkl"
    kg.save(kg_file)

    # Print stats
    print("\n=== Ingestion Complete ===")
    print(f"Total Documents: {len(files)}")
    print(f"Total Sections Extracted: {len(enriched_sections)}")
    print(f"Total Chunks Generated: {len(chunks)}")
    print(f"Knowledge Graph Stats: {kg.get_stats()}")
    print(f"Chunks saved to: {chunks_file}")
    print(f"Graph saved to: {kg_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full ingestion pipeline")
    parser.add_argument("--raw", type=str, default="data/raw", help="Path to raw data dir")
    parser.add_argument("--out", type=str, default="data/processed", help="Path to output processed dir")
    
    args = parser.parse_args()
    run_ingestion(Path(args.raw), Path(args.out))
