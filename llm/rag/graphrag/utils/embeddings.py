from __future__ import annotations
from typing import Iterable, List
import numpy as np


class Embeddings:
def __init__(self, provider: str, model: str):
self.provider = provider
self.model = model


def embed(self, texts: Iterable[str]) -> List[List[float]]:
texts = list(texts)
if self.provider == "ollama":
import ollama
out = ollama.embed(model=self.model, input=texts)
return out["embeddings"]
elif self.provider == "openai":
from openai import OpenAI
client = OpenAI()
out = client.embeddings.create(model=self.model, input=texts)
return [d.embedding for d in out.data]
elif self.provider == "azure":
from openai import AzureOpenAI
client = AzureOpenAI()
out = client.embeddings.create(model=self.model, input=texts)
return [d.embedding for d in out.data]
else:
raise ValueError(f"Unsupported embedding provider: {self.provider}")


@staticmethod
def mean_pool(vectors: List[List[float]]) -> List[float]:
arr = np.array(vectors)
return arr.mean(axis=0).tolist()
