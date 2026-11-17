
from vector_retriever import load_vector_store, retrieve_chunks


vector_store = load_vector_store()

questions = [

    "Explain Transformer architecture.",

    "How does self-attention work?",

    "What is the role of the encoder and decoder?",

    "Why did the authors eliminate recurrence?",

    "How does the Transformer perform machine translation?"

]


for question in questions:

    print("\n" + "=" * 70)

    print("QUESTION:", question)

    results = retrieve_chunks(
        vector_store=vector_store,
        query=question,
        top_k=5
    )

    for i, result in enumerate(results, start=1):

        print(f"\nRESULT {i}")

        print("Page:", result["page_number"])

        print("Chunk:", result["chunk_id"])

        print("Distance:", round(result["distance"], 4))

        print("Text:", result["text"][:600])
