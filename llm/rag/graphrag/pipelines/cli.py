from __future__ import annotations
import os
import yaml
import typer
from rich import print


from databases.neo4j_client import Neo4jClient, Neo4jConfig
from ..utils.embeddings import Embeddings
from ..utils.llm import LLM
from ..ingestion.document_loader import load_corpus
from ..ingestion.entity_extractor import extract_entities
from ..graph_builders.neo4j_builder import GraphBuilder
from ..retrievers.graph_walk import GraphRetriever
from ..prompting.system_prompts import ANSWER_SYSTEM


app = typer.Typer(add_completion=False)




def _load_cfg(path: str):
with open(path, "r", encoding="utf-8") as f:
return yaml.safe_load(f)




@app.command()
def ingest(config: str = typer.Option("configs/config.yaml")):
app()
