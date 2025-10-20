from __future__ import annotations
from typing import Dict, Iterable, List


from databases.neo4j_client import Neo4jClient
from ..utils.embeddings import Embeddings
from ..ingestion.document_loader import Chunk




class GraphBuilder:
def __init__(self, neo: Neo4jClient, embed: Embeddings):
self.neo = neo
self.embed = embed
self.neo.ensure_constraints()


def upsert_chunks(self, chunks: List[Chunk]):
texts = [c.text for c in chunks]
vecs = self.embed.embed(texts)
cypher = (
"UNWIND $rows AS row\n"
"MERGE (c:Chunk {id: row.id})\n"
"SET c.text = row.text, c.doc_id = row.doc_id, c.order = row.order, c.embedding = row.embedding"
)
rows = [dict(id=c.id, text=c.text, doc_id=c.doc_id, order=c.order, embedding=vecs[i]) for i, c in enumerate(chunks)]
self.neo.write(cypher, {"rows": rows})


def upsert_entities(self, chunk_id: str, payload: Dict):
ents = payload.get("entities", [])
rels = payload.get("relations", [])
cypher = (
"UNWIND $entities AS e\n"
"MERGE (en:Entity {id: e.id})\n"
"SET en.name = e.name, en.type = e.type\n"
"WITH 1 as _\n"
"UNWIND $rels AS r\n"
"MATCH (src:Entity {id: r.src}), (dst:Entity {id: r.dst})\n"
"MERGE (src)-[:RELATES {type: r.type}]->(dst)"
)
self.neo.write(cypher, {"entities": ents, "rels": rels})


# Link chunk mentions
cy2 = (
"MATCH (c:Chunk {id: $chunk_id})\n"
"UNWIND $entities AS e\n"
"MATCH (en:Entity {id: e.id})\n"
"MERGE (c)-[:MENTIONS]->(en)"
)
self.neo.write(cy2, {"chunk_id": chunk_id, "entities": ents})
