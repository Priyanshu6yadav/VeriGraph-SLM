"""
VeriGraph-SLM: Interactive Web Dashboard
----------------------------------------
A comprehensive, beginner-friendly Streamlit dashboard for:
1. Hybrid RAG Question Answering with Local SLM (qwen2.5:3b)
2. Formal Verification using Microsoft's Z3 SMT Solver
3. Interactive NetworkX Graph & Evidence Explorer (inspect nodes, edges, citations)
4. Upload & Ingest new PDF documents

To run:
  streamlit run app.py
"""

import sys
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Add 'src' folder to Python path
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import VeriGraph-SLM modules
from vector_retriever import load_vector_store
from graph_storage import load_graph, save_graph
from graph_builder import get_graph_statistics, add_triples, create_graph
from graph_visualizer import visualize_graph
from rag_pipeline import ask_question
from verifier import check_graph_consistency, verify_triple_with_z3
from main import ingest_document

# ----------------------------------------------------------------------
# Page Configuration & Styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="VeriGraph-SLM Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-verified {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-contradiction {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-unverified {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Helper: Load Cached Resources
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_stores():
    """Loads vector store and knowledge graph from disk into cache."""
    try:
        vs = load_vector_store()
    except Exception:
        vs = None

    try:
        g = load_graph()
    except Exception:
        g = None

    return vs, g


vector_store, graph = get_cached_stores()


# ----------------------------------------------------------------------
# Sidebar: System Status & PDF Ingestion
# ----------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("VeriGraph-SLM")
    st.markdown("**Hybrid GraphRAG + Z3 Formal Verification**")
    st.markdown("---")

    # System Status
    st.subheader("⚡ System Info")
    st.write("• **LLM**: `qwen2.5:3b` (Ollama)")
    st.write("• **Embeddings**: `qwen3-embedding:0.6b`")
    st.write("• **Verifier**: `Z3 SMT Solver`")

    if graph is not None:
        stats = get_graph_statistics(graph)
        st.write(f"• **Entities (Nodes)**: `{stats['total_entities']}`")
        st.write(f"• **Relations (Edges)**: `{stats['total_relationships']}`")
        st.write(f"• **Evidence Records**: `{stats['total_evidence_records']}`")
    else:
        st.warning("Knowledge Graph not loaded.")

    if vector_store is not None:
        st.write(f"• **Vector Chunks**: `{vector_store.index.ntotal}`")
    else:
        st.warning("Vector store not loaded.")

    st.markdown("---")

    # Ingestion Form
    st.subheader("📂 Ingest New PDF")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf"])
    max_c = st.slider("Max chunks for graph extraction", 2, 50, 10, help="Limit chunks for faster processing")

    if uploaded_file is not None:
        if st.button("🚀 Ingest Document", use_container_width=True):
            save_path = Path("data/raw_docs") / uploaded_file.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing PDF, embedding vectors, and extracting graph..."):
                ingest_document(str(save_path), max_chunks=max_c)
                st.cache_resource.clear()
                st.success("Ingestion complete! Please reload the page.")


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("<div class='main-title'>🛡️ VeriGraph-SLM Explorer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Grounded Question-Answering powered by Local SLMs, "
    "Knowledge Graphs, and Mathematical Verification via Z3 SMT Solver.</div>",
    unsafe_allow_html=True
)

# Tabs
tab_qa, tab_graph, tab_inspector, tab_verifier = st.tabs([
    "💬 Hybrid RAG Q&A",
    "🕸️ Interactive Graph View",
    "🔍 Node & Edge Inspector",
    "⚖️ Z3 Logic Verifier"
])


# ======================================================================
# TAB 1: Hybrid RAG Q&A with Z3 Verification
# ======================================================================
with tab_qa:
    st.subheader("Ask a Question with Hybrid Evidence")
    st.markdown("Combines **vector chunks** and **relational triples** with **Z3 mathematical verification**.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query_input = st.text_input(
            "Enter your question:",
            value="What does the Transformer architecture use and eliminate?",
            placeholder="e.g. How does Multi-Head Attention work?"
        )
    with col2:
        top_k = st.number_input("Top Evidence Chunks", min_value=1, max_value=10, value=4)

    # Sample query chips
    st.markdown("**Quick sample questions:**")
    c1, c2, c3 = st.columns(3)
    if c1.button("🔍 What does Transformer use?"):
        query_input = "What does the Transformer architecture use?"
    if c2.button("🔍 Explain Multi-Head Attention"):
        query_input = "What is Multi-Head Attention and how does it work?"
    if c3.button("🔍 Why eliminate recurrence?"):
        query_input = "Why did the authors eliminate recurrence in Transformer?"

    if st.button("✨ Ask VeriGraph-SLM", type="primary", use_container_width=True):
        if vector_store is None or graph is None:
            st.error("Vector Store or Knowledge Graph is not loaded. Please ingest a PDF first.")
        else:
            with st.spinner("Retrieving hybrid evidence & generating verified answer..."):
                result = ask_question(
                    query=query_input,
                    vector_store=vector_store,
                    graph=graph,
                    top_k=top_k
                )

            # Display Answer
            st.markdown("### 💡 Evidence-Grounded Answer")
            st.info(result["answer"])

            # Display Formally Verified Graph Facts
            st.markdown("### 🛡️ Formally Verified Graph Facts (Z3 SMT Solver)")
            verifications = result.get("verifications", [])
            if verifications:
                for v in verifications:
                    status = v["status"]
                    triple = v["triple"]
                    explanation = v["explanation"]

                    if status == "VERIFIED":
                        st.success(f"**[VERIFIED]** `{triple}`\n\n↳ {explanation}")
                    elif status == "CONTRADICTION":
                        st.error(f"**[CONTRADICTION]** `{triple}`\n\n↳ {explanation}")
                    else:
                        st.warning(f"**[UNVERIFIED]** `{triple}`\n\n↳ {explanation}")
            else:
                st.write("No direct graph relations matched.")

            # Display Document Citations
            st.markdown("### 📑 Supporting Document Evidence")
            sources = result.get("sources", [])
            for idx, src in enumerate(sources, start=1):
                methods = ", ".join(src.get("retrieval_method", []))
                with st.expander(f"Evidence {idx}: Page {src.get('page_number')} | Chunk {src.get('chunk_id')} (Found by: {methods})"):
                    st.write(src.get("text", "No text passage."))


# ======================================================================
# TAB 2: Interactive Knowledge Graph View (PyVis)
# ======================================================================
with tab_graph:
    st.subheader("Interactive Knowledge Graph Explorer")
    st.markdown("Explore entity connections, drag nodes, zoom in/out, and click edges to view citations.")

    html_path = Path("data/processed/knowledge_graph.html")
    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        components.html(html_content, height=750, scrolling=True)
    else:
        st.warning("Visualized graph HTML not found. Run ingestion first.")


# ======================================================================
# TAB 3: Interactive Node & Edge Inspector (Direct NetworkX Access)
# ======================================================================
with tab_inspector:
    st.subheader("Interactive NetworkX Graph Inspector")
    st.markdown("Directly explore and inspect any **Node (Entity)** or **Edge (Relationship)** in the graph.")

    if graph is None:
        st.warning("Knowledge Graph not loaded.")
    else:
        col_node, col_edge = st.columns(2)

        # --- Node Inspector ---
        with col_node:
            st.markdown("#### 🔵 Entity (Node) Inspector")
            all_nodes = sorted(list(graph.nodes()), key=lambda x: str(x).lower())

            selected_node = st.selectbox(
                "Select or search an entity:",
                options=all_nodes,
                index=all_nodes.index("Transformer") if "Transformer" in all_nodes else 0
            )

            if selected_node:
                in_degree = graph.in_degree(selected_node)
                out_degree = graph.out_degree(selected_node)
                total_deg = graph.degree(selected_node)

                st.markdown(f"**Entity:** `{selected_node}`")
                st.write(f"• **Total Connections:** `{total_deg}` (Out: `{out_degree}`, In: `{in_degree}`)")

                # Outgoing edges
                st.markdown("**Outgoing Relations (What it connects to):**")
                out_edges = graph.out_edges(selected_node, data=True)
                if out_edges:
                    for _, target, data in out_edges:
                        rel = data.get("relation", "CONNECTED_TO")
                        ev_count = len(data.get("evidence", []))
                        st.markdown(f"- `{selected_node}` --[**{rel}**]--> `{target}` *(Evidence: {ev_count})*")
                else:
                    st.caption("No outgoing relations.")

                # Incoming edges
                st.markdown("**Incoming Relations (What connects to it):**")
                in_edges = graph.in_edges(selected_node, data=True)
                if in_edges:
                    for source, _, data in in_edges:
                        rel = data.get("relation", "CONNECTED_TO")
                        ev_count = len(data.get("evidence", []))
                        st.markdown(f"- `{source}` --[**{rel}**]--> `{selected_node}` *(Evidence: {ev_count})*")
                else:
                    st.caption("No incoming relations.")

        # --- Edge & Evidence Inspector ---
        with col_edge:
            st.markdown("#### 🔗 Relationship (Edge) & Citation Inspector")
            all_edges_list = []
            for u, v, k, d in graph.edges(keys=True, data=True):
                all_edges_list.append((u, d.get("relation", k), v, d))

            edge_labels = [f"{u} --[{r}]--> {v}" for u, r, v, _ in all_edges_list]

            selected_edge_label = st.selectbox(
                "Select a relationship edge:",
                options=edge_labels,
                index=0 if edge_labels else None
            )

            if selected_edge_label and all_edges_list:
                chosen_idx = edge_labels.index(selected_edge_label)
                u, r, v, data = all_edges_list[chosen_idx]

                st.markdown(f"**Selected Relation:** `{u} --[{r}]--> {v}`")
                evidence_list = data.get("evidence", [])
                st.write(f"• **Number of Citation Records:** `{len(evidence_list)}`")

                for idx, ev in enumerate(evidence_list, start=1):
                    with st.expander(f"Citation {idx}: Page {ev.get('page_number')} (Chunk {ev.get('chunk_id')})"):
                        st.write(f"**Source Document:** `{ev.get('source', 'Unknown')}`")
                        st.write(f"**Exact Passage:**")
                        st.caption(ev.get("evidence_text", "No text recorded."))


# ======================================================================
# TAB 4: Z3 Logic Verifier & Contradiction Playground
# ======================================================================
with tab_verifier:
    st.subheader("⚖️ Z3 Logic Verifier & Contradiction Playground")
    st.markdown(
        "Mathematically verify claims or test hypothetical assertions against the Knowledge Graph "
        "using **Microsoft's Z3 SMT Solver**."
    )

    if graph is None:
        st.warning("Knowledge Graph not loaded.")
    else:
        # Full graph scan
        st.markdown("#### 1. Full Knowledge Graph Consistency Audit")
        if st.button("🔍 Run Full Graph Audit with Z3"):
            with st.spinner("Analyzing graph with Z3 SMT Solver..."):
                conflicts = check_graph_consistency(graph)
                if not conflicts:
                    st.success("✅ **PASS**: The Knowledge Graph is 100% logically consistent. No internal contradictions detected!")
                else:
                    st.error(f"⚠️ Found {len(conflicts)} logical contradictions:")
                    for c in conflicts:
                        st.write(f"• `{c['subject']}` has conflicting edges: `{c['relation_1']}` vs `{c['relation_2']}` with `{c['object']}`")

        st.markdown("---")

        # Interactive Claim Tester
        st.markdown("#### 2. Test a Custom Claim with Z3")
        st.caption("Enter any claim to test whether Z3 evaluates it as VERIFIED, CONTRADICTION, or UNVERIFIED.")

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            test_subj = st.text_input("Subject Entity", value="tensor2tensor")
        with vc2:
            test_pred = st.selectbox("Relation Predicate", ["USES", "REQUIRES", "ELIMINATES", "REPLACES", "CONTAINS", "GENERALIZES_TO"])
        with vc3:
            test_obj = st.text_input("Object Entity", value="earlier codebase")

        if st.button("⚡ Test Claim with Z3"):
            res = verify_triple_with_z3(test_subj, test_pred, test_obj, graph)
            st_status = res["status"]

            if st_status == "VERIFIED":
                st.success(f"### ✅ Status: VERIFIED\n\n{res['explanation']}")
            elif st_status == "CONTRADICTION":
                st.error(f"### ❌ Status: CONTRADICTION\n\n{res['explanation']}")
            else:
                st.warning(f"### ⚠️ Status: UNVERIFIED\n\n{res['explanation']}")

st.markdown("---")
st.caption("VeriGraph-SLM | Hybrid GraphRAG with Local SLMs and Formal Verification")
