from vector_store import create_embedding_model
from langchain_community.vectorstores import FAISS


def retrieve_chunks(
    vector_store,
    query: str,
    top_k: int = 5
):

    if not query.strip():
        raise ValueError("Query cannot be empty")

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k
    )

    retrieved_chunks = []

    for document, distance in results:

        retrieved_chunks.append({

            "text": document.page_content,

            "chunk_id": document.metadata["chunk_id"],

            "page_number": document.metadata["page_number"],

            "distance": float(distance)

        })

    return retrieved_chunks


def load_vector_store():

    embeddings = create_embedding_model()

    vector_store = FAISS.load_local(
        "data/processed/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store
