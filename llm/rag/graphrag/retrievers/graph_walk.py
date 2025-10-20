from __future__ import annotations

from typing import Dict, List, Tuple
import math

from databases.neo4j_client import Neo4jClient
from ..utils.embeddings import Embeddings


def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den_a = math.sqrt(sum(x * x for x in a))
    den_b = math.sqrt(sum(y * y for y in b))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


class GraphRetriever:
    """
    Hybrid GraphRAG retriever for Neo4j.

    - Uses vector index if available (db.index.vector.queryNodes)
    - Falls back to full-text index / substring search
    - Optionally expands across the entity graph to gather extra context
    - Re-ranks merged candidates with cosine(query_vec, candidate_embedding)
      when embeddings are present on nodes (c.embedding).
    """

    def __init__(
        self,
        neo: Neo4jClient,
        embed: Embeddings,
        top_k: int = 12,
        expand_hops: int = 2,
        alpha: float = 0.6,               # weight for vector score vs. text score
        vector_index: str = "chunk_embedding_idx",
        fulltext_index: str = "chunk_text_fts",
    ):
        self.neo = neo
        self.embed = embed
        self.top_k = top_k
        self.expand_hops = expand_hops
        self.alpha = alpha
        self.vector_index = vector_index
        self.fulltext_index = fulltext_index

    # ----------------------------- Index helpers -----------------------------

    def ensure_indexes(self, dim: int | None = None) -> None:
        """Create optional indexes when supported by the edition."""
        # Full-text index over chunk text
        try:
            self.neo.write(
                f"CREATE FULLTEXT INDEX {self.fulltext_index} IF NOT EXISTS "
                f"FOR (c:Chunk) ON EACH [c.text]"
            )
        except Exception:
            pass  # Community editions without procedure support, ignore.

        # Vector index (Neo4j 5+ with vector capability)
        if dim is not None:
            try:
                self.neo.create_vector_index(self.vector_index, dim)
            except Exception:
                pass

    # ------------------------- Candidate gathering --------------------------

    def _vector_candidates(self, qvec: List[float]) -> List[Dict]:
        cypher = (
            "CALL db.index.vector.queryNodes($name, $k, $q) YIELD node, score "
            "RETURN node.id AS id, node.text AS text, score "
            "ORDER BY score DESC"
        )
        try:
            return self.neo.run(
                cypher, {"name": self.vector_index, "k": self.top_k, "q": qvec}
            )
        except Exception:
            return []

    def _fulltext_candidates(self, qstr: str) -> List[Dict]:
        # Prefer full-text; fallback to substring search.
        cypher_ft = (
            "CALL db.index.fulltext.queryNodes($name, $q) YIELD node, score "
            "RETURN node.id AS id, node.text AS text, score "
            "ORDER BY score DESC LIMIT $k"
        )
        try:
            return self.neo.run(
                cypher_ft, {"name": self.fulltext_index, "q": qstr, "k": self.top_k}
            )
        except Exception:
            cypher_sub = (
                "MATCH (c:Chunk) "
                "WHERE c.text CONTAINS $q "
                "RETURN c.id AS id, c.text AS text, 1.0 AS score "
                "LIMIT $k"
            )
            return self.neo.run(cypher_sub, {"q": qstr, "k": self.top_k})

    # ------------------------- Merge & re-ranking ---------------------------

    def _fetch_embeddings(self, ids: List[str]) -> Dict[str, List[float]]:
        if not ids:
            return {}
        rows = self.neo.run(
            "MATCH (c:Chunk) WHERE c.id IN $ids "
            "RETURN c.id AS id, c.embedding AS emb",
            {"ids": ids},
        )
        return {r["id"]: r.get("emb") for r in rows if r.get("emb") is not None}

    def _normalize(self, hits: List[Dict], key: str = "score") -> Dict[str, float]:
        if not hits:
            return {}
        maxv = max(h[key] for h in hits) or 1.0
        return {h["id"]: (h[key] / maxv) for h in hits}

    def _merge_and_rerank(
        self, query_vec: List[float], vec_hits: List[Dict], ft_hits: List[Dict]
    ) -> List[Dict]:
        v_norm = self._normalize(vec_hits)
        f_norm = self._normalize(ft_hits)

        all_ids = list({*v_norm.keys(), *f_norm.keys()})
        id_to_text = {h["id"]: h["text"] for h in vec_hits + ft_hits}

        # Pull stored embeddings for cosine re
