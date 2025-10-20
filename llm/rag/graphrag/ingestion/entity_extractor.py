from __future__ import annotations
from typing import Dict, List
from .document_loader import Chunk
from ..utils.llm import LLM


ENTITY_PROMPT = (
"Extract named entities (people, orgs, locations, products, tickers) and key relations as triples.\n"
"Return JSON with fields 'entities' (list of {id,name,type}) and 'relations' (list of {src,dst,type}).\n"
)




def extract_entities(llm: LLM, chunk: Chunk) -> Dict:
messages = [
{"role": "system", "content": ENTITY_PROMPT},
{"role": "user", "content": chunk.text[:4000]},
]
raw = llm.chat(messages)
# Be forgiving: attempt to locate JSON in text
import json, re
m = re.search(r"\{[\s\S]*\}$", raw.strip())
if not m:
return {"entities": [], "relations": []}
try:
return json.loads(m.group(0))
except Exception:
return {"entities": [], "relations": []}
