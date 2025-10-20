from __future__ import annotations
from typing import List, Dict




def exact_match(pred: str, gold: str) -> float:
return float(pred.strip().lower() == gold.strip().lower())




def f1(pred_tokens: List[str], gold_tokens: List[str]) -> float:
ps, gs = set(pred_tokens), set(gold_tokens)
if not ps or not gs:
return 0.0
p = len(ps & gs) / len(ps)
r = len(ps & gs) / len(gs)
if p + r == 0:
return 0.0
return 2 * p * r / (p + r)
