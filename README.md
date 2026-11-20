# VeriGraph-SLM

**VeriGraph-SLM** is a local, privacy-first **Hybrid GraphRAG (Knowledge Graph + Vector Retrieval)** question-answering system powered by Small Language Models (SLMs) and formal verification.

---

## 🌟 Key Features

1. **Dual Retrieval**: Combines semantic vector search (**FAISS**) with relational reasoning (**NetworkX Knowledge Graph**).
2. **Evidence Traceability**: Every extracted relationship links back to its exact source PDF page and text chunk.
3. **100% Local**: Runs completely offline with **Ollama** (`qwen2.5:3b` and `qwen3-embedding:0.6b`). No API keys or cloud costs.
4. **Interactive Visualization**: Explores the knowledge graph visually in your browser with **PyVis**.
5. **Formal Verification (Planned)**: Uses **Z3 SMT Solver** to detect contradictions and verify facts.

---

## 🛠️ Prerequisites

1. **Python 3.11+** installed.
2. **Ollama** installed and running:
   ```bash
   # Download the LLM for reasoning and extraction
   ollama pull qwen2.5:3b

   # Download the embedding model for vector search
   ollama pull qwen3-embedding:0.6b
   ```

---

## 🚀 Quick Setup

1. **Clone the repository and create a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Run

### 1. Launch the Web Dashboard (Recommended)
```bash
streamlit run app.py
# or: python main.py --dashboard
```

### 2. Command Line Operations
```bash
# Ask a question via Hybrid RAG + Z3 Formal Verification
python main.py --query "What does the Transformer architecture use?"

# Audit the Knowledge Graph for contradictions using Z3
python main.py --verify-graph

# Ingest a new PDF document into FAISS and Knowledge Graph
python main.py --ingest "data/raw_docs/attention is you need.pdf"

# Open the interactive HTML Knowledge Graph
python main.py --visualize
```

---

## 📁 Project Structure

```
VeriGraph-SLM/
│
├── data/
│   ├── raw_docs/           # Input PDF documents (e.g., Attention Is All You Need)
│   └── processed/          # FAISS index and generated Knowledge Graph files
│
├── src/
│   ├── pdf_parser.py       # Extracts page-by-page text from PDFs
│   ├── text_cleaner.py     # Cleans and normalizes text
│   ├── chunker.py          # Splits text into overlapping chunks
│   ├── vector_store.py     # Embeds chunks and saves FAISS index
│   ├── vector_retriever.py # Retrieves top-k chunks via vector similarity
│   ├── graph_extractor.py  # Extracts (Subject-Predicate-Object) triples using SLM
│   ├── graph_builder.py    # Builds NetworkX MultiDiGraph with citations
│   ├── graph_storage.py    # Saves/loads the knowledge graph (.pkl)
│   ├── graph_retriever.py  # Retrieves subgraphs matching query entities
│   ├── hybrid_retriever.py # Fuses vector results + graph evidence
│   ├── graph_visualizer.py # Generates interactive HTML graph
│   └── rag_pipeline.py     # Question answering pipeline
│
├── requirements.txt        # Python package dependencies
└── pyproject.toml          # Project metadata & configuration
```
