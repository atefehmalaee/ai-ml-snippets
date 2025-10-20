from __future__ import annotations
from typing import Dict, List
import os


class LLM:
def __init__(self, provider: str, model: str):
self.provider = provider
self.model = model


def chat(self, messages: List[Dict[str, str]]) -> str:
# Minimal pluggable shim – extend with real SDKs
if self.provider == "ollama":
# pip install ollama
import ollama
rsp = ollama.chat(model=self.model, messages=messages)
return rsp["message"]["content"].strip()
elif self.provider == "openai":
from openai import OpenAI
client = OpenAI()
rsp = client.chat.completions.create(model=self.model, messages=messages)
return rsp.choices[0].message.content.strip()
elif self.provider == "azure":
from openai import AzureOpenAI
client = AzureOpenAI()
rsp = client.chat.completions.create(model=self.model, messages=messages)
return rsp.choices[0].message.content.strip()
else:
raise ValueError(f"Unsupported LLM provider: {self.provider}")
