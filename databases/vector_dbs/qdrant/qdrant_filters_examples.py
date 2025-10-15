"""
Qdrant Filter Examples
----------------------
Demonstrates all major Qdrant payload filter conditions:
- MatchText
- MatchValue
- MatchAny
- Range
- Geo filters
- Boolean logic (must / should / must_not)

Dependencies:
    pip install qdrant-client sentence-transformers
"""

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import numpy as np

# ==========================================
# 1️⃣ Connect to Qdrant
# ==========================================
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "filter_examples"

# Delete existing collection (safe reset)
if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
    client.delete_collection(COLLECTION_NAME)

# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
print(f"✅ Created collection '{COLLECTION_NAME}'")

# ==========================================
# 2️⃣ Insert sample payloads
# ==========================================
docs = [
    {
        "id": 1,
        "text": "LangChain integrates with Qdrant for LLM-powered retrieval.",
        "industry": "AI",
        "region": "APAC",
        "risk_score": 0.85,
        "created_at": 20240105,
        "location": {"lat": -33.86, "lon": 151.20},  # Sydney
    },
    {
        "id": 2,
        "text": "Qdrant is a vector database for finance and analytics use cases.",
        "industry": "Finance",
        "region": "EMEA",
        "risk_score": 0.65,
        "created_at": 20240210,
        "location": {"lat": 51.50, "lon": -0.12},  # London
    },
    {
        "id": 3,
        "text": "Open-source vector search enables semantic recommendation engines.",
        "industry": "E-commerce",
        "region": "Americas",
        "risk_score": 0.92,
        "created_at": 20230701,
        "location": {"lat": 40.71, "lon": -74.00},  # NYC
    },
]

# Embed using SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([d["text"] for d in docs])

# Upsert into Qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points=models.Batch(
        ids=[d["id"] for d in docs],
        vectors=embeddings.tolist(),
        payloads=docs,
    ),
)
print("✅ Inserted 3 documents")

# ==========================================
# 3️⃣ Build different filters
# ==========================================

# --- A. MatchText: text contains 'LLM' ---
filter_text = models.Filter(
    must=[
        models.FieldCondition(key="text", match=models.MatchText(text="LLM"))
    ]
)

# --- B. MatchValue: exact value ---
filter_value = models.Filter(
    must=[
        models.FieldCondition(key="industry", match=models.MatchValue(value="Finance"))
    ]
)

# --- C. MatchAny: field equals any in list ---
filter_any = models.Filter(
    must=[
        models.FieldCondition(key="region", match=models.MatchAny(any=["APAC", "Americas"]))
    ]
)

# --- D. Range: numeric range (risk_score between 0.7–0.9) ---
filter_range = models.Filter(
    must=[
        models.FieldCondition(key="risk_score", range=models.Range(gte=0.7, lte=0.9))
    ]
)

# --- E. GeoRadius: within 5000 km of Sydney ---
filter_geo = models.Filter(
    must=[
        models.FieldCondition(
            key="location",
            geo_radius=models.GeoRadius(
                center=models.GeoPoint(lat=-33.86, lon=151.20),
                radius=5_000_000  # in meters (~5000 km)
            )
        )
    ]
)

# --- F. Combined logic (AND + NOT) ---
filter_complex = models.Filter(
    must=[
        models.FieldCondition(key="industry", match=models.MatchAny(any=["AI", "Finance"])),
        models.FieldCondition(key="risk_score", range=models.Range(gte=0.6))
    ],
    must_not=[
        models.FieldCondition(key="region", match=models.MatchValue(value="EMEA"))
    ]
)

# ==========================================
# 4️⃣ Run searches with each filter
# ==========================================
query = "vector database for AI"
query_vector = model.encode([query])[0]

filters = {
    "MatchText (LLM)": filter_text,
    "MatchValue (Finance)": filter_value,
    "MatchAny (APAC/Americas)": filter_any,
    "Range (risk 0.7–0.9)": filter_range,
    "GeoRadius (near Sydney)": filter_geo,
    "Complex Logic (AI/Finance, not EMEA)": filter_complex,
}

for name, cond in filters.items():
    print(f"\n🔍 {name}")
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=cond,
        limit=3,
    )
    if not results:
        print("⚠️ No results found.")
    for r in results:
        print(f"• (score={r.score:.3f}) {r.payload['text']} [{r.payload['industry']}, {r.payload['region']}]")

# ==========================================
# 5️⃣ Clean up
# ==========================================
# client.delete_collection(COLLECTION_NAME)
# print(f"\n🧹 Deleted collection '{COLLECTION_NAME}'")
