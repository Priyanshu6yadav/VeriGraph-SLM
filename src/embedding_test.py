import numpy as np
from langchain_ollama import OllamaEmbeddings

# 1. Use 'model' instead of 'model_name'
model = OllamaEmbeddings(model="qwen3-embedding:0.6b")

sentences = [
    "Students must maintain 75% attendance.",
    "A minimum attendance of 75 percent is required.",
    "The Transformer uses attention mechanisms."
]

# 2. Use 'embed_documents' for a list of strings
embeddings = model.embed_documents(sentences)

# 3. Convert to a numpy array (or check len) to inspect shape
print("Embedding shape:", np.array(embeddings).shape)

for sentence, vector in zip(sentences, embeddings):
    print("\nSentence:", sentence)
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])