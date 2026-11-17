from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Embedding model running locally through ollama 
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


def create_embedding_model():
    embeddings = OllamaEmbeddings(
        model = EMBEDDING_MODEL
    )
    return embeddings

def create_vector_store(chunks: list):
    embeddings = create_embedding_model()

    documents = []

    for chunk in chunks:
        document = Document(
            page_content=chunk['text'],

            metadata = {
                "chunk_id": chunk['chunk_id'],
                "page_number": chunk['page_number']
            }
        )
        documents.append(document)


        vector_store = FAISS.from_documents(
            documents=documents,
            embedding = embeddings,
            normalize_L2 = True
        )
        return vector_store

def save_vector_store(vector_store):
    save_path = "data/processed/faiss_index"
    vector_store.save_local(save_path)
    print("Vector index saved:", save_path)

if __name__=="__main__":
    print("vector store module loaded successfully")
