from sentence_transformers import SentenceTransformer
import chromadb

# Connect to a local Chroma instance
client = chromadb.Client()

collection = client.create_collection("ai_snippets")

# Prepare data
texts = [
    "ChromaDB is a local vector database written in Python.",
    "It supports persistent storage and similarity search.",
    "Integration with LangChain makes RAG workflows easy."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# Insert documents
for i, text in enumerate(texts):
    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i].tolist()],
        documents=[text]
    )

# Search
query = "What is ChromaDB?"
query_vector = model.encode([query])[0]
results = collection.query(query_embeddings=[query_vector.tolist()], n_results=2)

print("🔍 Search results:", results)
