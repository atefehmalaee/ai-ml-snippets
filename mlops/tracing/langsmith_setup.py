"""
LangSmith Tracing Example — Hybrid Approach
-------------------------------------------

Demonstrates:
- @traceable decorator for high-level node tracing
- Manual sub-trace creation for inner calls (e.g. Qdrant, external API)
"""

import time
from langsmith import traceable, Client

# Initialize once per app
client = Client()

# Example: a Qdrant client placeholder
class DummyQdrantClient:
    def search(self, query):
        time.sleep(0.5)
        return ["Customer A", "Customer B", "Customer C"]

qdrant_client = DummyQdrantClient()


# ==========================================================
# 1️⃣  High-level trace (node)
# ==========================================================
@traceable(name="matcher_node")
def match_customers(opportunity_text: str):
    """
    Traced function that will appear as a top-level trace in LangSmith.
    Inside it, we manually instrument sub-traces for detailed metrics.
    """

    print("🔍 Matching customers for:", opportunity_text)

    # -------------------------------
    # Sub-trace: Qdrant vector search
    # -------------------------------
    qdrant_trace = client.create_trace(name="qdrant_search")
    start = time.time()

    matches = qdrant_client.search(query=opportunity_text)

    latency = time.time() - start
    client.log_metadata(qdrant_trace.id, {"latency_sec": latency})
    client.log_outputs(qdrant_trace.id, {"matches": matches})
    client.end_trace(qdrant_trace.id)

    print(f"✅ Found matches in {latency:.2f}s:", matches)
    return matches


# ==========================================================
# 2️⃣  Example entry point
# ==========================================================
if __name__ == "__main__":
    # Must have your LangSmith environment configured:
    # export LANGCHAIN_API_KEY=your_key
    # export LANGCHAIN_TRACING_V2=true

    results = match_customers("Fintech investment opportunity for retail clients")
    print("Final Result:", results)
