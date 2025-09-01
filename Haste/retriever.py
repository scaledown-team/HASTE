# SPDX-License-Identifier: MIT
# Hybrid retrieval: BM25 + OpenAI embeddings (no offline models, no caching).
# Includes BFS expansion over intra-module call edges and free-form query normalization.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import re
from rank_bm25 import BM25Okapi  # type: ignore
from openai import OpenAI  # type: ignore


# --- Query normalization ---


_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")
_SNAKE = re.compile(r"[_\-]")


def normalize_query(text: str) -> List[str]:
    text = text.strip()
    tokens: List[str] = []
    # Strip punctuation and parenthesis but keep underscores and dots, then split
    for raw in re.findall(r"[A-Za-z0-9_\.]+", text):
        raw = raw.lower()
        tokens.append(raw)
        # synonyms
        tokens.extend([p for p in raw.split(".") if p not in ("", raw)])
        # split snake/camel
        for part in _SNAKE.split(raw):
            if part and part != raw:
                tokens.append(part)
        camel_parts = _CAMEL.split(raw)
        for part in camel_parts:
            p = part.lower()
            if p and p != raw:
                tokens.append(p)
    # dedupe while preserving order
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --- Retrieval data model ---


@dataclass
class Doc:
    idx: int
    module: str
    qname: str
    kind: str
    name: str
    path: str
    docstring: str
    identifiers: List[str]
    signature: str
    start_byte: int
    end_byte: int
    score: Optional[float] = None


def build_bm25_corpus(docs: List[Doc]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized: List[List[str]] = []
    for d in docs:
        bag: List[str] = []
        bag.extend(normalize_query(d.name))
        bag.extend(normalize_query(d.qname))
        bag.extend(normalize_query(d.docstring))
        bag.extend([w.lower() for w in d.identifiers[:256]])  # cap
        bag.extend(normalize_query(d.signature))
        tokenized.append(bag)
    return BM25Okapi(tokenized), tokenized


def lexical_topk(docs: List[Doc], bm25: BM25Okapi, query: str, k: int, prefilter: int) -> List[Doc]:
    q_tokens = normalize_query(query)
    scores = bm25.get_scores(q_tokens)
    # take best prefilter, then keep best k after semantic re-rank
    idxs = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:prefilter]
    prelim = [Doc(**{**docs[i].__dict__, "score": float(scores[i])}) for i in idxs]
    return prelim[:k] if prefilter <= k else prelim


def _emb(client: OpenAI, model: str, text: str) -> List[float]:
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding  # type: ignore


def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) + 1e-12
    db = math.sqrt(sum(x * x for x in b)) + 1e-12
    return num / (da * db)


def semantic_rerank(prelim: List[Doc],
                    query: str,
                    model: str,
                    src_bytes: Optional[bytes] = None,
                    src_by_path: Optional[Dict[str, bytes]] = None) -> List[Doc]:
    # Only call OpenAI if we actually need to (prelim>0)
    if not prelim:
        return []
    client = OpenAI()
    q_emb = _emb(client, model, query)
    # doc text for embedding: name + signature + docstring + first ~200 chars of body
    out: List[Doc] = []
    for d in prelim:
        body_preview = ""
        if src_by_path is not None:
            b = src_by_path.get(d.path)
            if b is not None:
                body_preview = b[d.start_byte:min(d.end_byte, d.start_byte + 1200)].decode("utf-8", "ignore")
        elif src_bytes is not None:
            body_preview = src_bytes[d.start_byte:min(d.end_byte, d.start_byte + 1200)].decode("utf-8", "ignore")
        text = f"{d.qname}\n{d.signature}\n{d.docstring}\n{body_preview}"
        e = _emb(client, model, text)
        s = _cos(q_emb, e)
        out.append(Doc(**{**d.__dict__, "score": 0.5 * float(d.score or 0.0) + 0.5 * s}))
    out.sort(key=lambda x: x.score or 0.0, reverse=True)
    return out


# --- BFS expansion on same-module call edges ---


def bfs_expand(selected: List[Doc],
               docs_by_name: Dict[str, List[Doc]],
               call_edges: Dict[str, List[str]],
               depth: int,
               max_add: int) -> List[Doc]:
    keep: Dict[str, Doc] = {d.qname: d for d in selected}
    frontier = [d.qname for d in selected]
    d = 0
    while frontier and d < depth and len(keep) < (len(selected) + max_add):
        nxt: List[str] = []
        for q in frontier:
            for callee in call_edges.get(q, []):
                # callee is unqualified; look up by simple name
                cand = docs_by_name.get(callee, [])
                for c in cand:
                    if c.qname not in keep:
                        keep[c.qname] = c
                        nxt.append(c.qname)
                        if len(keep) >= (len(selected) + max_add):
                            break
        frontier = nxt
        d += 1
    return list(keep.values())


