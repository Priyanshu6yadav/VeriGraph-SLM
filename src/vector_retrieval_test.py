from ingestion_test import process_pdf

from vector_store import (
    create_vector_store,
    save_vector_store
)

from vector_retriever import retrieve_chunks


# Step 1: Process PDF
chunks = process_pdf(
    "data/raw_docs/attention is you need.pdf"
)

print("Total chunks:", len(chunks))


# Step 2: Create FAISS vector store
vector_store = create_vector_store(chunks)

print("FAISS index created successfully!")


# Step 3: Save vector store
save_vector_store(vector_store)


# Step 4: Ask a question
query = "How does the Transformer eliminate recurrence?"


# Step 5: Retrieve relevant chunks
results = retrieve_chunks(
    vector_store=vector_store,
    query=query,
    top_k=5
)


# Step 6: Display results
for result in results:

    print("\n" + "=" * 60)

    print("Chunk:", result["chunk_id"])

    print("Page:", result["page_number"])

    print("Distance:", round(result["distance"], 4))

    print("Content:", result["text"])
