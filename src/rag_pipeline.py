"""
VeriGraph-SLM: Hybrid RAG Pipeline
----------------------------------
This module integrates both:
1. FAISS Vector Search (semantic similarity)
2. NetworkX Knowledge Graph (entity-relationship facts)

It passes both the factual relationships and document passages into a local
Small Language Model (qwen2.5:3b via Ollama) to generate evidence-backed answers.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import retrievers, storage loaders, and formal verifier
from vector_retriever import load_vector_store
from graph_storage import load_graph
from hybrid_retriever import hybrid_retrieve
from verifier import verify_triple_with_z3


# ----------------------------------------------------------------------
# 1. Initialize Local Language Model (SLM)
# ----------------------------------------------------------------------
# We use qwen2.5:3b running locally through Ollama with temperature=0
# for factual and deterministic responses.
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# ----------------------------------------------------------------------
# 2. Define Evidence-Grounded Prompt Template
# ----------------------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """You are an evidence-based document question-answering assistant.

Answer the user's question using ONLY the supplied Knowledge Graph Facts and Document Passages.

Rules:
1. Do not invent information or make assumptions.
2. Rely strictly on the provided facts and passages.
3. If the context does not contain the answer, state: "The document does not provide sufficient information to answer this question."
4. Be concise, clear, and cite supporting relationships or pages where relevant.

==================== KNOWLEDGE GRAPH FACTS ====================
{graph_facts}

==================== DOCUMENT PASSAGES ====================
{document_passages}

USER QUESTION: {question}

Provide your evidence-grounded answer:
"""
)

# Build the LangChain pipeline: Prompt -> LLM -> String Output
rag_chain = prompt | llm | StrOutputParser()


# ----------------------------------------------------------------------
# 3. Format Context for Prompt (Graph Facts + Passages)
# ----------------------------------------------------------------------
def format_hybrid_context(hybrid_results, max_passages: int = 5):
    """
    Formats the output from hybrid_retrieve() into structured text
    suitable for the LLM prompt.
    """
    # 1. Format Knowledge Graph relationships
    relationships = hybrid_results.get("relationships", [])
    if relationships:
        graph_facts_list = []
        for rel in relationships[:10]:  # Limit to top 10 relevant relationships
            graph_facts_list.append(
                f"- {rel['subject']} --[{rel['relation']}]--> {rel['object']}"
            )
        graph_facts_text = "\n".join(graph_facts_list)
    else:
        graph_facts_text = "No direct knowledge graph facts matched."

    # 2. Format Document Passages from combined evidence
    combined = hybrid_results.get("combined_evidence", [])
    if not combined:
        passages_text = "No document passages found."
    else:
        # Prioritize evidence found by BOTH methods, then by vector
        sorted_evidence = sorted(
            combined,
            key=lambda x: (
                len(x.get("retrieval_method", [])),  # Found by both vector & graph first
                -(x.get("distance") or 999.0)        # Lower distance is better
            ),
            reverse=True
        )

        passage_parts = []
        for index, item in enumerate(sorted_evidence[:max_passages], start=1):
            methods = ", ".join(item.get("retrieval_method", []))
            passage_parts.append(
                f"[{index}] Source: {item.get('source', 'Unknown')} | "
                f"Page: {item.get('page_number', 'N/A')} | "
                f"Chunk ID: {item.get('chunk_id', 'N/A')} | "
                f"Retrieved by: {methods}\n"
                f"Content: {item.get('text', '').strip()}"
            )
        passages_text = "\n\n".join(passage_parts)

    return graph_facts_text, passages_text


# ----------------------------------------------------------------------
# 4. Main Question Answering Function
# ----------------------------------------------------------------------
def ask_question(query: str, vector_store, graph, top_k: int = 5):
    """
    Executes the complete Hybrid RAG workflow:
    1. Retrieves evidence using vector search + knowledge graph.
    2. Formats both relation triples and text passages.
    3. Invokes the SLM to produce an evidence-backed answer.
    """
    # Step 1: Perform Hybrid Retrieval
    hybrid_results = hybrid_retrieve(
        query=query,
        vector_store=vector_store,
        graph=graph,
        top_k=top_k
    )

    combined_evidence = hybrid_results.get("combined_evidence", [])
    if not combined_evidence and not hybrid_results.get("relationships"):
        return {
            "answer": "No relevant document evidence or graph facts were found.",
            "relationships": [],
            "sources": []
        }

    # Step 2: Format context for prompt
    graph_facts, document_passages = format_hybrid_context(
        hybrid_results=hybrid_results,
        max_passages=top_k
    )

    # Step 3: Generate answer using the SLM
    answer = rag_chain.invoke({
        "graph_facts": graph_facts,
        "document_passages": document_passages,
        "question": query
    })

    # Step 4: Formally verify the top supporting graph relationships with Z3
    verified_relationships = []
    for rel in hybrid_results.get("relationships", [])[:5]:
        v_check = verify_triple_with_z3(
            subject=rel["subject"],
            relation=rel["relation"],
            object_=rel["object"],
            graph=graph
        )
        verified_relationships.append({
            "triple": f"{rel['subject']} --[{rel['relation']}]--> {rel['object']}",
            "status": v_check["status"],
            "verified": v_check["verified"],
            "explanation": v_check["explanation"]
        })

    return {
        "answer": answer.strip(),
        "matched_entities": hybrid_results.get("matched_entities", []),
        "relationships": hybrid_results.get("relationships", [])[:5],
        "verifications": verified_relationships,
        "sources": combined_evidence[:top_k]
    }


# ----------------------------------------------------------------------
# 5. Interactive Testing Loop
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Loading Vector Store and Knowledge Graph ---")
    vector_store = load_vector_store()
    graph = load_graph()
    print("Ready for Hybrid RAG Queries!\n")

    test_questions = [
        "What does the Transformer architecture use and eliminate?",
    ]

    for question in test_questions:
        print("=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        result = ask_question(
            query=question,
            vector_store=vector_store,
            graph=graph,
            top_k=4
        )

        print("\n--- ANSWER ---")
        print(result["answer"])

        print("\n--- SUPPORTING GRAPH RELATIONS (Z3 VERIFIED) ---")
        for v in result.get("verifications", []):
            badge = "✅ [VERIFIED]" if v["verified"] else f"⚠️ [{v['status']}]"
            print(f"{badge} {v['triple']}")
            print(f"   Reason: {v['explanation']}")

        print("\n--- TOP SUPPORTING EVIDENCE ---")
        for src in result["sources"][:3]:
            methods = ", ".join(src.get("retrieval_method", []))
            print(f"Page {src.get('page_number')} (Chunk {src.get('chunk_id')}) [Found by: {methods}]")
