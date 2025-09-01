#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics and scoring utilities: PageRank, identifier TF-IDF, structure richness,
API influence boost, feature scoring, and BM25 for query-aware reranking.
"""

import math
import re
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

from .index import RepoIndex, NodeRecord, is_public_function_name, node_loc, cyclomatic_complexity
from .identifiers import collect_identifiers
from .ts_utils import walk

def pagerank_on_calls(idx: RepoIndex, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 100) -> Dict[str, float]:
    nodes = list(idx.functions.keys())
    if not nodes:
        return {}

    symbols_to_qnames: DefaultDict[str, Set[str]] = defaultdict(set)
    for q, rec in idx.functions.items():
        symbols_to_qnames[rec.name].add(q)

    out_edges: Dict[str, Dict[str, float]] = {q: {} for q in nodes}
    for caller, callee_counts in idx.calls.items():
        out_edges.setdefault(caller, {})
        for callee_sym, cnt in callee_counts.items():
            for target_q in symbols_to_qnames.get(callee_sym, ()):  # resolve by symbol name
                if caller == target_q:
                    continue
                out_edges[caller][target_q] = out_edges[caller].get(target_q, 0.0) + float(cnt)

    N = len(nodes)
    rank = {q: 1.0 / N for q in nodes}
    for _ in range(max_iter):
        new_rank = {q: (1.0 - alpha) / N for q in nodes}
        for u in nodes:
            outs = out_edges.get(u, {})
            if outs:
                total_w = sum(outs.values()) or 1.0
                share = alpha * rank[u]
                for v, w in outs.items():
                    new_rank[v] += share * (w / total_w)
            else:
                share = alpha * (rank[u] / N)
                for v in nodes:
                    new_rank[v] += share
        delta = sum(abs(new_rank[q] - rank[q]) for q in nodes)
        rank = new_rank
        if delta < tol:
            break
    return rank


def compute_identifier_idf(idx: RepoIndex) -> Dict[str, float]:
    df: Counter = Counter()
    total_docs = max(1, len(idx.functions))
    for rec in idx.functions.values():
        ids = set(collect_identifiers(rec.node, rec.src_bytes, filtered=True))
        for t in ids:
            df[t] += 1
    return {t: math.log((total_docs + 1) / (dfc + 1)) + 1.0 for t, dfc in df.items()}


def function_identifier_tfidf(rec: NodeRecord, idf: Dict[str, float]) -> float:
    toks = collect_identifiers(rec.node, rec.src_bytes, filtered=True)
    if not toks:
        return 0.0
    tf = Counter(toks)
    total = float(len(toks))
    return sum((c / total) * idf.get(t, 1.0) for t, c in tf.items())


def structure_richness(node) -> float:
    seen: Set[Tuple[int, str]] = set()
    def dfs(n, d):
        seen.add((d, n.type))
        for c in n.children:
            dfs(c, d+1)
    dfs(node, 0)
    size = max(1, sum(1 for _ in walk(node)))
    return min(1.0, len(seen) / (math.log2(size + 1) + 5.0))


def api_influence_boost(rec: NodeRecord, idx: RepoIndex) -> float:
    info = idx.module_api_info.get(rec.path, {})
    all_exports = info.get("__all__", set())
    main_calls = info.get("main_calls", set())
    boost = 0.0
    parts = rec.qname.split("::")
    is_top_level_fn = (rec.type == "function" and len(parts) == 2 and is_public_function_name(rec.name))
    if is_top_level_fn:
        boost += 0.2
    if rec.name in all_exports:
        boost += 0.2
    if rec.name in main_calls:
        boost += 0.2
    return min(boost, 0.5)


def normalize_component(x: float, cap: float) -> float:
    return min(1.0, x / cap)


def score_features(feat: Dict[str, float]) -> float:
    loc_n   = normalize_component(feat["loc"],   200.0)
    cc_n    = normalize_component(feat["cc"],     20.0)
    pr_n    = max(0.0, min(1.0, feat["pr"]))
    tfidf_n = normalize_component(feat["tfidf"],   2.0)
    struct  = max(0.0, min(1.0, feat["struct"]))
    api_b   = max(0.0, min(0.5, feat["api"])) * 2.0
    w_cc, w_loc, w_pr, w_tfidf, w_struct, w_api = 0.25, 0.15, 0.25, 0.15, 0.15, 0.05
    base = (w_cc*cc_n + w_loc*loc_n + w_pr*pr_n + w_tfidf*tfidf_n + w_struct*struct + w_api*api_b)
    return round(base, 4)


def compute_raw_features(
    rec: NodeRecord,
    pr_norm: Dict[str, float],
    idf: Dict[str, float],
    idx: RepoIndex,
) -> Dict[str, float]:
    """Assemble intrinsic features for a function record.

    - loc: lines of code (node span)
    - cc: cyclomatic complexity
    - pr: PageRank value (normalized)
    - tfidf: identifier TF-IDF
    - struct: structure richness
    - api: public API / entrypoint influence
    """
    return {
        "loc": node_loc(rec.node),
        "cc": cyclomatic_complexity(rec.node),
        "pr": pr_norm.get(rec.qname, 0.0),
        "tfidf": function_identifier_tfidf(rec, idf),
        "struct": structure_richness(rec.node),
        "api": api_influence_boost(rec, idx),
    }


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tok(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())


def bm25(query: str, docs: Dict[str, str], k1=1.2, b=0.75) -> Dict[str, float]:
    q_terms = _tok(query)
    if not q_terms:
        return {k: 0.0 for k in docs}
    N = len(docs) or 1
    doc_toks = {k: _tok(v) for k, v in docs.items()}
    avgdl = sum(len(v) for v in doc_toks.values()) / max(1, N)
    df = Counter(t for toks in doc_toks.values() for t in set(toks))
    idf = {t: math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0) for t in set(q_terms)}
    scores: Dict[str, float] = {}
    for k, toks in doc_toks.items():
        tf = Counter(toks); dl = len(toks) or 1
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            s += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[k] = s
    m = max(scores.values()) if scores else 1.0
    return {k: (v / m if m > 0 else 0.0) for k, v in scores.items()}


