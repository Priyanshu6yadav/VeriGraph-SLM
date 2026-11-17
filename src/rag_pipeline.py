from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector_retriever import (
    load_vector_store,
    retrieve_chunks
)


# ------------------------------------------------
# 1. Initialize local language model
# ------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# ------------------------------------------------
# 2. Create prompt template
# ------------------------------------------------

prompt = ChatPromptTemplate.from_template(

    """
        You are an evidence-based document question-answering assistant.

        Answer the user's question using ONLY the supplied document context.

        Rules:
        1. Do not invent information.
        2. Do not use outside knowledge to fill missing information.
        3. If the context does not contain the answer, clearly say that
        the document does not provide sufficient information.
        4. Explain the answer clearly.
        5. Preserve important conditions and exceptions.

        DOCUMENT CONTEXT:{context}
        USER QUESTION:{question}

        Provide your answer:
    """
)


# ------------------------------------------------
# 3. Create LangChain chain
# ------------------------------------------------

rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)


# ------------------------------------------------
# 4. Format retrieved documents
# ------------------------------------------------

def format_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
                Source: {result.get('source', 'Unknown document')}
                Page: {result.get('page_number', 'Unknown')}
                Chunk ID: {result.get('chunk_id', 'Unknown')}
                Content:{result.get('text', '')}
            """
        )

    return "\n\n".join(context_parts)



# ------------------------------------------------
# 5. Main RAG function
# ------------------------------------------------

def ask_question(query, vector_store, top_k=5):

    # Retrieve relevant chunks
    results = retrieve_chunks(
        vector_store=vector_store,
        query=query,
        top_k=top_k
    )

    if not results:
        return {
            "answer": "No relevant document evidence was retrieved.",
            "sources": []
        }

    # Prepare context
    context = format_context(results)

    # Generate answer
    answer = rag_chain.invoke({

        "context": context,

        "question": query

    })

    return {

        "answer": answer,

        "sources": results

    }


# ------------------------------------------------
# 6. Test application
# ------------------------------------------------

if __name__ == "__main__":

    vector_store = load_vector_store()

    while True:

        question = input("\nAsk a question (type exit to stop): ")

        if question.lower() == "exit":
            break

        result = ask_question(
            question,
            vector_store
        )

        print("\nANSWER:\n")
        print(result["answer"])

        print("\nRETRIEVED SOURCES:\n")

        for source in result["sources"]:

            print(
                f"Page: {source['page_number']} | "
                f"Chunk: {source['chunk_id']} | "
                f"Distance: {source['distance']:.4f}"
            )
