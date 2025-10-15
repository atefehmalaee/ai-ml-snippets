"""
Pinecone: create index → upsert vectors → query.

Dependencies:
    pip install pinecone-client sentence-transformers
"""

import pinecone
from sentence_transformers import SentenceTransformer
import os

# -----------------------------
# 1. Connect to Pinecone
# -----------------------------
# You must set your API key:
# export PINECONE_API_KEY="your-key"
api_key = os.getenv("PINECONE_API_KEY")

pinecone.init(api_key=api_key, environment="gcp-starter")

INDEX_NAME = "ai-snippets-index"

# Delete existing index if it exists
if INDEX_NAME in pinecone.list_indexes():
    pinecone.delete_index(INDEX_NAME)

# Create a new index
pinecone.create_index(name=INDEX_NAME, dimension=384, metric="cosine")
index = pinecone.Index(INDEX_NAME)
print(f"✅ Created and connected to index '{INDEX_NAME}'")

# -----------------------------
# 2. Prepare sample data
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [
    "Pinecone is a managed vector database service.",
    "It supports fast and scalable semantic search.",
    "Vector databases are essential for LLM retrieval."
]
vectors = model.encode(texts)

# -----------------------------
# 3. Upsert vectors
# -----------------------------
to_upsert = [(str(i), vectors[i].tolist(), {"text": texts[i]}) for i in range(len(texts))]
index.upsert(vectors=to_upsert)
print("✅ Inserted 3 sample vectors")

# -----------------------------
# 4. Query
# -----------------------------
query = "What does Pinecone do?"
query_vector = model.encode([query])[0].tolist()

results = index.query(vector=query_vector, top_k=2, include_metadata=True)

print("\n🔍 Top matches:")
for match in results["matches"]:
    print(f"• {match['metadata']['text']} (score={match['score']:.3f})")

# -----------------------------
# 5. Cleanup
# -----------------------------
# pinecone.delete_index(INDEX_NAME)
