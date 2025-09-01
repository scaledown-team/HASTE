#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrieval and fusion utilities for selecting root functions.

Features:
- BM25 document building (name + signature + identifiers + body prefix)
- Optional semantic retrieval via SentenceTransformers
- Rank-based fusion (RRF) across available channels
- Lexical floor guarantee for BM25 roots
"""

from __future__ import annotations
from sentence_transformers import SentenceTransformer  # type: ignore
import numpy as np  # type: ignore
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable
import math
import re
from collections import defaultdict

from .index import RepoIndex
from .ts_utils import ts_node_text
from .identifiers import collect_identifiers
from .metrics import bm25
from .index import py_function_signature
from .query_utils import expand_query


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tok(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())


def build_bm25_docs(idx: RepoIndex) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    for qn, rec in idx.functions.items():
        sig = py_function_signature(rec.node, rec.src_bytes)
        ids = " ".join(collect_identifiers(rec.node, rec.src_bytes, filtered=True))
        body = ts_node_text(rec.src_bytes, rec.node)
        btoks = _tok(body)[:120]
        docs[qn] = f"{rec.name} {sig} {ids} " + " ".join(btoks)
    return docs


def maybe_dense_scores(query: str, docs: Dict[str, str], model_name: str) -> Optional[Dict[str, float]]:
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        return None

    keys = list(docs.keys())
    corpus = [docs[k] for k in keys]
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    d_emb = model.encode(corpus, normalize_embeddings=True, convert_to_numpy=True)
    sims = (d_emb @ q_emb)
    sims = (sims - sims.min()) / float((sims.max() - sims.min()) or 1.0)
    return {keys[i]: float(sims[i]) for i in range(len(keys))}


def rrf_fuse(rankings: Dict[str, List[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for _src, lst in rankings.items():
        for i, q in enumerate(lst):
            scores[q] += 1.0 / (k + (i + 1))
    return dict(scores)


def choose_roots(
    idx: RepoIndex,
    base_scores: Dict[str, float],
    query: Optional[str],
    top_k: int,
    *,
    lexical_min_pct: float = 0.5,
    rrf_k: int = 60,
    use_semantic: bool = False,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[str]:
    if top_k <= 0:
        return []

    # Structural rank (fallback and a channel for fusion)
    struct_rank = [q for q, _ in sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)]

    if not query:
        return struct_rank[:top_k]

    docs = build_bm25_docs(idx)
    bm25_scores = bm25(query, docs)
    bm25_rank = sorted(bm25_scores.keys(), key=lambda q: bm25_scores[q], reverse=True)

    rankings: Dict[str, List[str]] = {"bm25": bm25_rank, "struct": struct_rank}

    if use_semantic:
        dense = maybe_dense_scores(query, docs, semantic_model_name)
        if dense:
            dense_rank = sorted(dense.keys(), key=lambda q: dense[q], reverse=True)
            rankings["dense"] = dense_rank

    fused = rrf_fuse(rankings, k=max(1, int(rrf_k)))
    fused_rank = sorted(fused.keys(), key=lambda q: fused[q], reverse=True)

    # Lexical floor
    keep_lex = max(0, min(top_k, math.ceil(top_k * max(0.0, min(1.0, lexical_min_pct)))))
    roots: List[str] = bm25_rank[:keep_lex]
    for q in fused_rank:
        if len(roots) >= top_k:
            break
        if q not in roots:
            roots.append(q)
    return roots


def hybrid_scores(
    idx: RepoIndex,
    query: Optional[str],
    *,
    rrf_k: int = 60,
    use_semantic: bool = False,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[Dict[str, float], Optional[float], Optional[float]]:
    """Return fused RRF scores per function qname, plus top-1 BM25 and dense scores for gating.

    If no query is provided, returns empty scores and Nones.
    """
    if not query:
        return {}, None, None

    docs = build_bm25_docs(idx)
    # query expansion for better lexical recall
    q_expanded = " ".join(expand_query(query)) if query else ""
    bm25_scores = bm25(q_expanded or (query or ""), docs)
    bm25_rank = sorted(bm25_scores.keys(), key=lambda q: bm25_scores[q], reverse=True)

    rankings: Dict[str, List[str]] = {"bm25": bm25_rank}
    top1_bm25 = bm25_scores[bm25_rank[0]] if bm25_rank else None

    top1_dense: Optional[float] = None
    if use_semantic:
        dense = maybe_dense_scores(query, docs, semantic_model_name)
        if dense:
            dense_rank = sorted(dense.keys(), key=lambda q: dense[q], reverse=True)
            rankings["dense"] = dense_rank
            top1_dense = dense[dense_rank[0]] if dense_rank else None

    fused = rrf_fuse(rankings, k=max(1, int(rrf_k))) if rankings else {}
    # normalize fused to [0,1]
    if fused:
        m = max(fused.values()) or 1.0
        fused = {k: v / m for k, v in fused.items()}
    return fused, top1_bm25, top1_dense


def choose_under_token_budget(
    idx: RepoIndex,
    base_scores: Dict[str, float],
    fused_scores: Dict[str, float],
    token_budget: int,
    *,
    lambda_mix: float = 0.85,
) -> List[str]:
    """Greedy knapsack: benefit per token. Benefit = λ*fused + (1-λ)*normalized_static.

    Token cost is approximated by lexical token count of the function snippet.
    """
    if token_budget <= 0:
        return []
    # normalize base scores to [0,1]
    if base_scores:
        m = max(base_scores.values()) or 1.0
        base_norm = {k: v / m for k, v in base_scores.items()}
    else:
        base_norm = {}

    # Precompute per-item cost and benefit
    items: List[Tuple[str, float, int]] = []  # (qname, benefit, cost)
    for qn, rec in idx.functions.items():
        code = ts_node_text(rec.src_bytes, rec.node)
        cost = max(1, len(_tok(code)))
        fused = fused_scores.get(qn, 0.0)
        static = base_norm.get(qn, 0.0)
        benefit = float(max(0.0, min(1.0, lambda_mix))) * fused + (1.0 - float(max(0.0, min(1.0, lambda_mix)))) * static
        items.append((qn, benefit, cost))

    # Greedy by benefit-per-cost, with tie-breakers on higher benefit then lower cost
    items.sort(key=lambda x: (x[1] / x[2], x[1], -x[2]), reverse=True)
    selected: List[str] = []
    budget_left = token_budget
    for qn, benefit, cost in items:
        if cost <= budget_left:
            selected.append(qn)
            budget_left -= cost
        if budget_left <= 0:
            break
    return selected


def should_retrieve(
    query: Optional[str],
    top1_bm25: Optional[float],
    top1_dense: Optional[float],
    *,
    bm25_thresh: float = 0.05,
    dense_thresh: float = 0.05,
) -> bool:
    if not query:
        return False
    b_ok = (top1_bm25 is not None and top1_bm25 >= bm25_thresh)
    d_ok = (top1_dense is not None and top1_dense >= dense_thresh)
    return b_ok or d_ok


def lexical_coverage(text: str, query_terms: List[str]) -> float:
    if not query_terms:
        return 0.0
    lower = text.lower()
    hits = sum(1 for t in query_terms if t and t.lower() in lower)
    return hits / max(1, len(query_terms))


def select_seeds_and_expand(
    idx: RepoIndex,
    fused_scores: Dict[str, float],
    *,
    top_k: int,
    depth: int,
    lexical_min_pct: float = 0.0,
    query: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Pick seed qnames by fused score (with optional lexical gating), then expand by directional BFS.

    Returns (seeds, final_ordered_qnames). Both lists are hard-capped to top_k.
    """
    if top_k <= 0:
        return [], []

    query_terms: List[str] = []
    if query:
        query_terms = re.findall(r"\w+", query.lower())

    # 1) sort by fused scores
    ranked = [q for q, _ in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)]

    # 2) lexical gating on seeds (optional)
    seeds: List[str] = []
    for qn in ranked:
        if len(seeds) >= top_k:
            break
        if lexical_min_pct > 0.0 and query_terms:
            rec = idx.functions.get(qn)
            if rec is None:
                continue
            text = ts_node_text(rec.src_bytes, rec.node)
            if lexical_coverage(text, query_terms) < lexical_min_pct:
                continue
        seeds.append(qn)

    # 3) Expand by BFS over call graph up to `depth`, include callers and callees
    def neighbors(qname: str) -> Iterable[str]:
        out: Set[str] = set()
        # outgoing: qname -> called symbols
        for callee_sym in idx.calls.get(qname, {}):
            for target_q in (f for f in idx.functions if f.endswith(f"::{callee_sym}")):
                if target_q != qname:
                    out.add(target_q)
        # incoming: any caller that references our symbol
        self_sym = idx.functions[qname].name
        for caller, callee_counts in idx.calls.items():
            if self_sym in callee_counts and caller != qname:
                out.add(caller)
        return out

    seen: Set[str] = set(seeds)
    order: List[str] = list(seeds)
    frontier: List[str] = list(seeds)
    for _ in range(max(0, depth)):
        nxt: List[str] = []
        for u in frontier:
            for v in neighbors(u):
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
        order.extend(nxt)

    # 4) Re-rank expanded set by fused score for stability, then cap to top_k
    order = sorted(order, key=lambda q: fused_scores.get(q, 0.0), reverse=True)[:top_k]

    return seeds, order


