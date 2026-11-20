from vector_store import create_embedding_model
from langchain_community.vectorstores import FAISS



def retrieve_chunks(
    vector_store,
    query: str,
    top_k: int = 5
):

    if not query or not query.strip():
        return []

    # Qwen3-Embedding retrieval instruction
    instructed_query = (
        "Instruct: Given a question, retrieve relevant "
        "passages from the document that answer the question.\n"
        f"Query: {query}"
    )

    # Check how many vectors are actually indexed
    total_vectors = vector_store.index.ntotal

    if total_vectors == 0:
        return []

    results = vector_store.similarity_search_with_score(
        query=instructed_query,
        k=min(top_k, total_vectors)
    )

    retrieved_chunks = []

    for document, distance in results:

        retrieved_chunks.append({

            "text": document.page_content,

            "chunk_id": document.metadata["chunk_id"],

            "page_number": document.metadata["page_number"],

            "source": document.metadata.get(
                "source", "Unknown document"
            ),

            "distance": float(distance)

        })

    print(f"\nTotal indexed vectors: {total_vectors}")
    print(f"Retrieved chunks: {len(retrieved_chunks)}")

    return retrieved_chunks



def load_vector_store():

    embeddings = create_embedding_model()

    vector_store = FAISS.load_local(
        "data/processed/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store
