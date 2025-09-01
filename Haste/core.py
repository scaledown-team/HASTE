#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core hybrid selection and payload assembly with strict top_k enforcement.

Implements:
- Lexical + semantic (optional) fusion via RRF, combined with intrinsic base score
- Seed selection capped at top_k, optional lexical gating
- Directional BFS expansion over call graph, then re-rank by fused and cap to top_k
- Code blob only when include_code is True and includes only selected functions
"""

from __future__ import annotations
from .embeddings import OpenAIEmbedder
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set
import re

from .index import RepoIndex, NodeRecord, py_function_signature, extract_docstring_if_first_stmt
from .payload import function_entry, class_entry
from .metrics import compute_identifier_idf, compute_raw_features, score_features
from .query_utils import expand_query
from .ts_utils import ts_node_text


def _tokenize(q: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", q or "")] 


def _lexical_score(text: str, q_tokens: Sequence[str]) -> float:
    if not q_tokens:
        return 0.0
    text_l = (text or "").lower()
    hits = sum(1 for t in set(q_tokens) if t and t in text_l)
    return hits / float(len(set(q_tokens)))


def _rrf_from_ranked(ranked_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for r, qn in enumerate(lst):
            scores[qn] = scores.get(qn, 0.0) + 1.0 / (k + r + 1)
    return scores


def _rank_ids_by_score(scores: Dict[str, float]) -> List[str]:
    return sorted(scores.keys(), key=lambda k: (-scores[k], k))


def _unique(seq: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _neighbors(idx: RepoIndex, qname: str) -> Set[str]:
    # both directions: callers and callees; resolve by symbol name for callees
    out: Set[str] = set()
    # outgoing (callee symbols → qnames)
    callee_counts = idx.calls.get(qname, {})
    for callee_sym in callee_counts.keys():
        for target_q in (f for f in idx.functions if f.endswith(f"::{callee_sym}")):
            if target_q != qname:
                out.add(target_q)
    # incoming (callers referencing our symbol)
    self_sym = idx.functions[qname].name
    for caller, cc in idx.calls.items():
        if self_sym in cc and caller != qname:
            out.add(caller)
    return out


def _function_text_for_query(rec: NodeRecord) -> str:
    sig = py_function_signature(rec.node, rec.src_bytes)
    doc = extract_docstring_if_first_stmt(rec.node.child_by_field_name("body"), rec.src_bytes) or ""
    return f"{rec.qname} {sig} {doc}"


def select_roots_and_expand(
    idx: RepoIndex,
    base_scores: Dict[str, float],
    fused_query_scores: Dict[str, float],
    *,
    query: Optional[str],
    query_weight: float,
    lexical_min_pct: float,
    lex_rank_order: List[str],
    depth: int,
    top_k: int,
) -> Tuple[List[str], List[str]]:
    # Combine fused query score with intrinsic base score
    final_scores: Dict[str, float] = {}
    qw = max(0.0, min(1.0, float(query_weight)))
    for qn in idx.functions.keys():
        final_scores[qn] = qw * fused_query_scores.get(qn, 0.0) + (1.0 - qw) * base_scores.get(qn, 0.0)

    # Lexical gate
    q_tokens = _tokenize(" ".join(expand_query(query or "")))
    def passes_gate(qn: str) -> bool:
        if lexical_min_pct <= 0.0 or not q_tokens:
            return True
        text = _function_text_for_query(idx.functions[qn])
        return _lexical_score(text, q_tokens) >= lexical_min_pct

    # Seeds: apply lexical floor then fill from fused ranking, cap to top_k
    ranked_all = sorted(final_scores.keys(), key=lambda k: (-final_scores[k], k))
    seeds: List[str] = []
    need_lex = max(0, min(top_k, int(round(lexical_min_pct * top_k))))
    if need_lex > 0 and lex_rank_order:
        for qn in lex_rank_order:
            if len(seeds) >= need_lex:
                break
            if passes_gate(qn) and qn not in seeds:
                seeds.append(qn)
    for qn in ranked_all:
        if not passes_gate(qn):
            continue
        if qn in seeds:
            continue
        seeds.append(qn)
        if len(seeds) >= max(0, top_k):
            break
    if not seeds:
        seeds = ranked_all[:max(0, top_k)]

    # BFS expansion with fused preference, capped to top_k
    seen: Set[str] = set(seeds)
    order: List[str] = []
    q = deque(seeds)
    while q and len(order) < top_k:
        cur = q.popleft()
        order.append(cur)
        if depth <= 0:
            continue
        neigh_sorted = sorted(_neighbors(idx, cur), key=lambda n: (-final_scores.get(n, 0.0), n))
        for v in neigh_sorted:
            if v not in seen and (len(order) + len(q)) < top_k:
                seen.add(v)
                q.append(v)

    order = _unique(order)[:max(0, top_k)]
    seeds = seeds[:max(0, top_k)]
    return seeds, order


def build_llm_payload_hybrid(
    idx: RepoIndex,
    page_rank_norm: Dict[str, float],
    idf: Dict[str, float],
    *,
    include_code: bool,
    top_k: int,
    depth: int,
    query: Optional[str],
    query_weight: float,
    lexical_min_pct: float,
    rrf_k: int = 60,
    max_tokens: Optional[int] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    # 1) base scores (intrinsic) per function
    base_scores: Dict[str, float] = {}
    for qn, rec in idx.functions.items():
        feats = compute_raw_features(rec, page_rank_norm, idf, idx)
        base_scores[qn] = score_features(feats)

    # 2) fused query scores via RRF (lexical + optional semantic)
    texts: Dict[str, str] = {qn: _function_text_for_query(rec) for qn, rec in idx.functions.items()}
    q_tokens = _tokenize(" ".join(expand_query(query or "")))

    # lexical ranking via BM25 over rich docs
    from .retrieval import build_bm25_docs
    from .metrics import bm25
    docs = build_bm25_docs(idx)
    q_string = " ".join(expand_query(query or ""))
    bm25_scores = bm25(q_string, docs) if q_string else {k: 0.0 for k in docs}
    lex_rank = [qn for qn, _ in sorted(bm25_scores.items(), key=lambda kv: kv[1], reverse=True)]

    # semantic ranking (optional): not wired by default; if embedder provided, score over function text
    sem_rank: List[str] = []
    if embedder is not None and query:
        try:
            import numpy as np  # type: ignore
            qv = embedder.encode([query])[0]
            sims: Dict[str, float] = {}
            for qn in texts.keys():
                dv = embedder.encode([texts[qn]])[0]
                num = float(np.dot(qv, dv))
                den = float(np.linalg.norm(qv) * np.linalg.norm(dv) + 1e-12)
                sims[qn] = num / den
            sem_rank = _rank_ids_by_score(sims)
        except Exception:
            sem_rank = []

    rrf_scores = _rrf_from_ranked([lex_rank] + ([sem_rank] if sem_rank else []), k=rrf_k)

    # 3) select seeds and expand, strict caps
    seeds, final_qnames = select_roots_and_expand(
        idx,
        base_scores=base_scores,
        fused_query_scores=rrf_scores,
        query=query,
        query_weight=query_weight,
        lexical_min_pct=lexical_min_pct,
        lex_rank_order=lex_rank,
        depth=depth,
        top_k=top_k,
    )

    # 4) emit nodes strictly limited to top_k, ordered by fused score combiner
    final_scores: Dict[str, float] = {}
    qw = max(0.0, min(1.0, float(query_weight)))
    for qn in final_qnames:
        final_scores[qn] = qw * rrf_scores.get(qn, 0.0) + (1.0 - qw) * base_scores.get(qn, 0.0)
    final_qnames = sorted(final_qnames, key=lambda qn: (-final_scores.get(qn, 0.0), qn))[:max(0, top_k)]

    nodes = []
    for qn in final_qnames:
        rec = idx.functions[qn]
        nodes.append({**function_entry(rec, final_scores.get(qn, 0.0), include_code=False)})

    # Minimal classes: include classes from same modules as selected functions (small cap)
    sel_modules = {idx.functions[qn].module for qn in final_qnames}
    classes = [class_entry(rec, include_code=False)
               for _qn, rec in idx.classes.items() if rec.module in sel_modules][:10]

    out: Dict[str, Any] = {
        "summary": {"total_functions": len(idx.functions), "total_classes": len(idx.classes)},
        "nodes": nodes,
        "classes": classes,
        "selected": {
            "roots": seeds,
            "functions": final_qnames,
            "classes": [c["qname"] for c in classes],
        },
    }

    if include_code:
        # deterministic pack up to max_tokens (approx 4 chars/token)
        parts: List[str] = []
        remaining_chars = None if max_tokens is None else int(max_tokens) * 4
        for qn in final_qnames:
            rec = idx.functions[qn]
            snippet = ts_node_text(rec.src_bytes, rec.node)
            if remaining_chars is None:
                parts.append(snippet)
            else:
                if len(snippet) <= remaining_chars:
                    parts.append(snippet)
                    remaining_chars -= len(snippet)
                else:
                    break
        out["code"] = "\n".join(parts)

    return out


