from __future__ import annotations
import os
import uuid
from pathlib import Path
from typing import Dict, Iterable, List


from pydantic import BaseModel




class Chunk(BaseModel):
id: str
doc_id: str
text: str
order: int




def _read_text(path: Path) -> str:
if path.suffix.lower() in {".txt", ".md"}:
return path.read_text(encoding="utf-8", errors="ignore")
# Simple PDF fallback (no heavy deps):
if path.suffix.lower() == ".pdf":
try:
import pypdf
reader = pypdf.PdfReader(str(path))
return "\n".join(page.extract_text() or "" for page in reader.pages)
except Exception:
return "" # keep lightweight here
return ""




def simple_chunk(text: str, size: int = 800, overlap: int = 120) -> List[str]:
words = text.split()
chunks = []
start = 0
while start < len(words):
end = min(len(words), start + size)
chunks.append(" ".join(words[start:end]))
if end == len(words):
return chunks
