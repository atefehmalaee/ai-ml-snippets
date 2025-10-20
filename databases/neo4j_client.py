from __future__ import annotations
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable, TransientError




DEFAULT_DB = os.getenv("NEO4J_DATABASE", "neo4j")




def _to_dict(record) -> Dict[str, Any]:
# Flatten a neo4j.Record into a regular dict
d = {}
for k in record.keys():
v = record[k]
try:
d[k] = dict(v)
except Exception:
d[k] = v
return d




@dataclass
class Neo4jConfig:
uri: str
user: str
password: str
database: str = DEFAULT_DB
max_retries: int = 4
retry_backoff: float = 0.5


@staticmethod
def from_env(prefix: str = "NEO4J_") -> "Neo4jConfig":
uri = os.getenv(f"{prefix}URI") or os.getenv(f"{prefix}URL")
if not uri:
# support Aura style bolt+s routing
uri = "bolt://localhost:7687"
return Neo4jConfig(
uri=uri,
user=os.getenv(f"{prefix}USER", "neo4j"),
password=os.getenv(f"{prefix}PASSWORD", "password"),
database=os.getenv(f"{prefix}DATABASE", DEFAULT_DB),
)




class Neo4jClient:
def __init__(self, cfg: Neo4jConfig):
self.cfg = cfg
self._driver = GraphDatabase.driver(cfg.uri, auth=basic_auth(cfg.user, cfg.password))


@classmethod
def from_env(cls, prefix: str = "NEO4J_") -> "Neo4jClient":
pass
