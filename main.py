"""
VeriGraph-SLM: Unified Command-Line Interface (CLI)
---------------------------------------------------
This is the main entrypoint for the entire project.

It provides simple, single-command operations for:
1. Ingesting new PDFs (Vector Index + Knowledge Graph)
2. Querying the system with Hybrid RAG + Z3 Formal Verification
3. Checking internal graph consistency
4. Generating and viewing the interactive Knowledge Graph

Usage Examples:
  python main.py --query "What does the Transformer architecture use?"
  python main.py --ingest "data/raw_docs/attention is you need.pdf" --max-chunks 10
  python main.py --visualize
  python main.py --verify-graph
"""

import sys
import argparse
import webbrowser
from pathlib import Path

# Add 'src' folder to Python path so we can import modules directly
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import project modules
from pdf_parser import extract_pdf_text
from text_cleaner import clean_text
from chunker import create_chunks
from vector_store import create_vector_store, save_vector_store
from vector_retriever import load_vector_store
from graph_builder import create_graph, add_triples, get_graph_statistics
from graph_storage import save_graph, load_graph
from graph_extractor import extract_triples_from_chunk
from graph_visualizer import visualize_graph
from rag_pipeline import ask_question
from verifier import check_graph_consistency


# ----------------------------------------------------------------------
# 1. Ingestion Pipeline: PDF -> Vector Index + Knowledge Graph
# ----------------------------------------------------------------------
def ingest_document(pdf_path: str, max_chunks: int = None):
    """
    Automates the full ingestion process for a PDF:
    - Step 1: Extracts text from PDF pages
    - Step 2: Cleans and splits text into overlapping chunks
    - Step 3: Embeds chunks and saves the FAISS vector index
    - Step 4: Extracts factual triples using SLM (Ollama)
    - Step 5: Builds, saves, and visualizes the Knowledge Graph
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF file '{pdf_file}' not found.")
        return

    print(f"\n==================================================")
    print(f"🚀 INGESTING DOCUMENT: {pdf_file.name}")
    print(f"==================================================")

    # Step 1: Extract pages
    print("\n[1/5] Extracting text from PDF...")
    pages = extract_pdf_text(str(pdf_file))
    print(f"  -> Extracted {len(pages)} pages.")

    # Step 2: Clean and chunk text
    print("\n[2/5] Cleaning and chunking text...")
    all_chunks = []
    source_name = pdf_file.name

    for page in pages:
        cleaned = clean_text(page["text"])
        chunks = create_chunks(cleaned, chunk_size=500, overlap=100)
        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"page_{page['page_number']}_chunk_{idx}",
                "page_number": page["page_number"],
                "text": chunk_text,
                "source": source_name
            })

    total_chunks = len(all_chunks)
    print(f"  -> Created {total_chunks} total text chunks.")

    # Step 3: Create & save FAISS Vector Store
    print("\n[3/5] Generating dense embeddings & saving FAISS index...")
    vector_store = create_vector_store(all_chunks)
    save_vector_store(vector_store)
    print("  -> FAISS index saved to 'data/processed/faiss_index'.")

    # Step 4: Extract Knowledge Graph Triples
    print("\n[4/5] Extracting Knowledge Graph triples via local SLM (qwen2.5:3b)...")
    chunks_to_process = all_chunks[:max_chunks] if max_chunks else all_chunks
    print(f"  -> Processing {len(chunks_to_process)} chunks for graph extraction...")

    # Load existing graph if present, otherwise create new
    try:
        graph = load_graph()
        print("  -> Loaded existing graph to append new knowledge.")
    except Exception:
        graph = create_graph()
        print("  -> Initialized a new Knowledge Graph.")

    all_triples = []
    for i, chunk in enumerate(chunks_to_process, start=1):
        print(f"     Extracting triples [{i}/{len(chunks_to_process)}] - {chunk['chunk_id']}...", end="\r")
        triples = extract_triples_from_chunk(chunk)
        all_triples.extend(triples)

    print(f"\n  -> Extracted {len(all_triples)} raw relational triples.")

    # Step 5: Build, Save & Visualize Knowledge Graph
    print("\n[5/5] Building and saving Knowledge Graph...")
    graph = add_triples(graph, all_triples)
    save_graph(graph)
    stats = get_graph_statistics(graph)
    print(f"  -> Graph Stats: {stats['total_entities']} Entities, {stats['total_relationships']} Relations, {stats['total_evidence_records']} Evidence records.")

    # Generate interactive HTML
    visualize_graph(graph)
    print("\n✅ Ingestion complete! FAISS index and Knowledge Graph are ready.")


# ----------------------------------------------------------------------
# 2. Query Pipeline: Hybrid RAG + Z3 Verification
# ----------------------------------------------------------------------
def query_system(question: str, top_k: int = 5):
    """
    Loads stores and runs Hybrid RAG + Z3 Formal Verification on a user question.
    """
    print(f"\n==================================================")
    print(f"🔍 QUESTION: {question}")
    print(f"==================================================")

    # Load stored FAISS index and Knowledge Graph
    try:
        vector_store = load_vector_store()
        graph = load_graph()
    except Exception as e:
        print(f"Error loading indexes: {e}")
        print("Please ingest a document first using: python main.py --ingest <pdf_path>")
        return

    # Execute Hybrid RAG + Verification
    result = ask_question(
        query=question,
        vector_store=vector_store,
        graph=graph,
        top_k=top_k
    )

    print("\n" + "=" * 50)
    print("💡 ANSWER:")
    print("=" * 50)
    print(result["answer"])

    print("\n" + "=" * 50)
    print("🛡️ FORMALLY VERIFIED GRAPH FACTS (Z3 SMT SOLVER):")
    print("=" * 50)
    verifications = result.get("verifications", [])
    if verifications:
        for v in verifications:
            badge = "✅ [VERIFIED]" if v["verified"] else f"⚠️ [{v['status']}]"
            print(f"{badge} {v['triple']}")
            print(f"   ↳ {v['explanation']}")
    else:
        print("No direct graph relations matched for this query.")

    print("\n" + "=" * 50)
    print("📑 TOP SUPPORTING DOCUMENT EVIDENCE:")
    print("=" * 50)
    for idx, src in enumerate(result.get("sources", [])[:3], start=1):
        methods = ", ".join(src.get("retrieval_method", []))
        print(f"[{idx}] Page {src.get('page_number')} | Chunk: {src.get('chunk_id')} | Found by: {methods}")
        print(f"    Excerpt: {src.get('text', '')[:200]}...\n")


# ----------------------------------------------------------------------
# 3. Main CLI Argument Parser
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VeriGraph-SLM: Hybrid GraphRAG with Local SLMs and Formal Verification"
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Ask a question using Hybrid RAG + Z3 Formal Verification"
    )
    parser.add_argument(
        "--ingest", "-i",
        type=str,
        help="Path to PDF file to ingest into Vector Store and Knowledge Graph"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunks to process for LLM graph extraction (useful for quick testing)"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Open the interactive Knowledge Graph HTML in default browser"
    )
    parser.add_argument(
        "--verify-graph",
        action="store_true",
        help="Check the Knowledge Graph for internal logical contradictions using Z3"
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Launch the interactive Streamlit Web Dashboard"
    )

    args = parser.parse_args()

    # Route based on flags provided
    if args.dashboard:
        import subprocess
        app_file = Path(__file__).resolve().parent / "app.py"
        print(f"Launching VeriGraph-SLM Dashboard: streamlit run {app_file.name} ...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file)])

    elif args.query:
        query_system(args.query)

    elif args.ingest:
        ingest_document(args.ingest, max_chunks=args.max_chunks)

    elif args.verify_graph:
        print("\n--- Scanning Knowledge Graph with Z3 SMT Solver ---")
        graph = load_graph()
        conflicts = check_graph_consistency(graph)
        if not conflicts:
            print("✅ PASS: The Knowledge Graph is 100% logically consistent (No internal contradictions detected).")
        else:
            print(f"⚠️ Found {len(conflicts)} logical contradictions:")
            for c in conflicts:
                print(f" - {c['subject']} conflicting relations: {c['relation_1']} vs {c['relation_2']}")

    elif args.visualize:
        html_file = Path("data/processed/knowledge_graph.html")
        if html_file.exists():
            print(f"Opening Knowledge Graph: {html_file.resolve()}")
            webbrowser.open(str(html_file.resolve()))
        else:
            print("Knowledge graph HTML not found. Run --ingest first to generate it.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
