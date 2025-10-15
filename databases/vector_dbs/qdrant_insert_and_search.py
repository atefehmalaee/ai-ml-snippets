"""
Qdrant example: connect → create collection → insert vectors →
perform similarity search → apply filtering → delete → clean up.

Dependencies:
    pip install qdrant-client sentence-transformers
"""

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import numpy as np


# -----------------------------
# 1. Connect to Qdrant
# -----------------------------
# For local Docker instance: Qdrant runs on http://localhost:6333 by default
client = QdrantClient(host="localhost", port=6333)

# -----------------------------
# 2. Define collection schema
# -----------------------------
COLLECTION_NAME = "ai_snippets"

# Delete collection if it exists (safe reset)
if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
    client.delete_collection(COLLECTION_NAME)

# Create a new collection with vector size = 384 (e.g., MiniLM embeddings)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

print(f"✅ Created collection '{COLLECTION_NAME}'")

# -----------------------------
# 3. Prepare example documents
# -----------------------------
docs = [
    {"id": 1, "text": "Qdrant is an open-source vector database."},
    {"id": 2, "text": "Vector search enables semantic similarity queries."},
    {"id": 3, "text": "LangChain and Qdrant integrate for LLM-powered retrieval."}
]

# Use SentenceTransformer to create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([d["text"] for d in docs])

# -----------------------------
# 4. Insert (Upsert) documents
# -----------------------------
client.upsert(
    collection_name=COLLECTION_NAME,
    points=models.Batch(
        ids=[d["id"] for d in docs],
        vectors=embeddings.tolist(),
        payloads=[{"text": d["text"]} for d in docs]
    )
)

print("✅ Inserted 3 documents into Qdrant")

# -----------------------------
# 5. Perform similarity search
# -----------------------------
query = "What is Qdrant?"
query_vector = model.encode([query])[0]

search_result = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=2
)

print("\n🔍 Search Results:")
for r in search_result:
    print(f"• (Score={r.score:.3f}) {r.payload['text']}")

# -----------------------------
# 6. Filtered search example
# -----------------------------
# Filter payloads that contain 'LLM'
filter_condition = models.Filter(
    must=[
        models.FieldCondition(
            key="text",
            match=models.MatchText(text="LLM")
        )
    ]
)

filtered_result = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    query_filter=filter_condition,
    limit=2
)

print("\n🎯 Filtered Search (LLM-related):")
for r in filtered_result:
    print(f"• {r.payload['text']} (score={r.score:.3f})")

# -----------------------------
# 7. Delete example
# -----------------------------
client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=models.PointIdsList(points=[1])
)
print("\n🗑️ Deleted point with ID 1")

# -----------------------------
# 8. Cleanup (optional)
# -----------------------------
# client.delete_collection(COLLECTION_NAME)
# print(f"🧹 Deleted collection '{COLLECTION_NAME}'")
