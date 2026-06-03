import os
import json
from pathlib import Path
from tqdm import tqdm

from src.ingestion.parsers import parse_document
from src.ingestion.parsers.pdf_parser import parse_multiple_pdfs
from src.ingestion.metadata_enricher import enrich_section
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.kg_builder import SectionKnowledgeGraph
from src.ingestion.embedder import TelecomEmbedder
from src.ingestion.indexer import QdrantIndexer

def run_ingestion(raw_data_dir: Path, output_dir: Path):
    print("=== Starting End-to-End Ingestion ===")
    
    # Ensure dirs exist
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_3gpp_dir = raw_data_dir / "3gpp"
    
    if not raw_3gpp_dir.exists():
        print(f"Directory {raw_3gpp_dir} not found. Ensure data is downloaded.")
        return

    # Find all PDFs and HTMLs
    files = list(raw_3gpp_dir.glob("*.pdf")) + list(raw_3gpp_dir.glob("*.html")) + list(raw_3gpp_dir.glob("*.txt"))
    if not files:
        print("No documents found to parse.")
        return

    print(f"Found {len(files)} documents.")
    
    all_sections = []
    
    # Step 2: Parse documents (parallel for PDFs, sequential for HTML/TXT)
    pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
    html_files = [f for f in files if f.suffix.lower() in ['.html', '.htm']]
    txt_files = [f for f in files if f.suffix.lower() == '.txt']
    
    if pdf_files:
        print(f"Parsing {len(pdf_files)} PDFs in parallel (ThreadPoolExecutor)...")
        pdf_sections = parse_multiple_pdfs(pdf_files, max_workers=4)
        all_sections.extend(pdf_sections)
    
    if html_files:
        print(f"Parsing {len(html_files)} HTML files...")
        for file_path in tqdm(html_files):
            sections = parse_document(file_path)
            all_sections.extend(sections)

    if txt_files:
        print(f"Parsing {len(txt_files)} text files...")
        for file_path in tqdm(txt_files):
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

    # Step 6: Embed and Index
    print("Embedding chunks...")
    embedder = TelecomEmbedder(batch_size=32)
    embedded_chunks = embedder.embed_documents(chunks)

    print("Indexing to Qdrant...")
    indexer = QdrantIndexer()
    indexer.index_chunks(embedded_chunks)

    # Step 7: Save outputs
    print("Saving outputs...")
    chunks_file = output_dir / "chunks.jsonl"
    with open(chunks_file, "w") as f:
        for chunk in chunks:
            # Flatten metadata into the chunk for saving
            safe_chunk = {k: v for k, v in chunk.items() if k != "metadata"}
            metadata = chunk.get("metadata", {})
            safe_chunk["spec_number"] = metadata.get("spec_number")
            safe_chunk["clause_string"] = metadata.get("clause_string")
            safe_chunk["clause_title"] = metadata.get("clause_title")
            safe_chunk["clause_path"] = metadata.get("clause_path", [])
            safe_chunk["cross_references"] = metadata.get("cross_references", [])
            f.write(json.dumps(safe_chunk, default=str) + "\n")
            
    kg_file = output_dir / "section_graph.pkl"
    kg.save(kg_file)

    # Print stats
    print("\n=== Ingestion Complete ===")
    print(f"Total Documents: {len(files)}")
    print(f"Total Sections Extracted: {len(enriched_sections)}")
    print(f"Total Chunks Generated: {len(chunks)}")
    print(f"Knowledge Graph Stats: {kg.get_stats()}")
    print(f"Total Chunks Indexed: {len(embedded_chunks)}")
    print(f"Chunks saved to: {chunks_file}")
    print(f"Graph saved to: {kg_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full ingestion pipeline")
    parser.add_argument("--raw", type=str, default="data/raw", help="Path to raw data dir")
    parser.add_argument("--out", type=str, default="data/processed", help="Path to output processed dir")
    
    args = parser.parse_args()
    run_ingestion(Path(args.raw), Path(args.out))
