from dataclasses import dataclass
from typing import List
import re
import numpy as np


@dataclass
class Scored:
    idx: int
    score: float


def _tokenize_field(s: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9_]+", (s or "").lower()) if t]


def build_bm25(nodes):
    # Lazy import to avoid hard dependency at import time
    from rank_bm25 import BM25Okapi  # type: ignore
    docs = []
    for n in nodes:
        bag = " ".join([
            " ".join(n.identifiers),
            n.docstring or "",
            n.comments or "",
            n.name or "",
        ])
        docs.append(_tokenize_field(bag))
    return BM25Okapi(docs)


def lexical_prefilter(nodes, query_tokens: List[str], top_m: int) -> List[Scored]:
    bm25 = build_bm25(nodes)
    scores = bm25.get_scores(query_tokens)
    order = np.argsort(scores)[::-1][:max(0, int(top_m))]
    return [Scored(int(i), float(scores[int(i)])) for i in order]


def semantic_rerank(nodes, cand: List[Scored], q_embed, d_embeds) -> List[Scored]:
    q = np.array(q_embed, dtype=np.float32)
    sims = []
    for emb in d_embeds:
        v = np.array(emb, dtype=np.float32)
        denom = (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9)
        sims.append(float(np.dot(q, v) / denom))
    lex = np.array([c.score for c in cand], dtype=np.float32)
    if lex.size and float(lex.max()) > 0:
        lex = lex / (float(lex.max()) + 1e-9)
    sem = np.array(sims, dtype=np.float32)
    if sem.size and float(sem.max()) > 0:
        sem = sem / (float(sem.max()) + 1e-9)
    fused = 0.5 * lex + 0.5 * sem
    order = np.argsort(fused)[::-1]
    return [Scored(cand[int(i)].idx, float(fused[int(i)])) for i in order]


