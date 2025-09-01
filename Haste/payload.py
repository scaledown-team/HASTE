#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM payload assembly and query-aware ranking orchestration.
"""

from typing import Any, Dict, List, Optional, Tuple

from .index import RepoIndex, NodeRecord, extract_docstring_if_first_stmt, py_function_signature, class_bases_text
from .metrics import (
    pagerank_on_calls,
    compute_identifier_idf,
    function_identifier_tfidf,
    structure_richness,
    api_influence_boost,
    score_features,
    bm25,
)
from .selection import slice_from_roots, assemble
from .retrieval import select_seeds_and_expand
from .index import cyclomatic_complexity, node_loc

def function_entry(rec: NodeRecord, score: float, include_code: bool=False) -> Dict[str, Any]:
    src = rec.src_bytes
    node = rec.node
    sig = py_function_signature(node, src)
    doc = extract_docstring_if_first_stmt(node.child_by_field_name("body"), src)
    entry = {
        "type": rec.type,
        "name": rec.name,
        "qname": rec.qname,
        "module": rec.module,
        "path": str(rec.path),
        "lineno": rec.start,
        "end_lineno": rec.end,
        "signature": sig,
        "docstring": doc,
        "score": score,
    }
    if include_code:
        from .ts_utils import ts_node_text
        entry["code"] = ts_node_text(src, node)
    return entry


def class_entry(rec: NodeRecord, include_code: bool=False) -> Dict[str, Any]:
    src = rec.src_bytes
    node = rec.node
    bases = class_bases_text(node, src)
    doc = extract_docstring_if_first_stmt(node.child_by_field_name("body"), src)
    e = {
        "type": rec.type,
        "name": rec.name,
        "qname": rec.qname,
        "module": rec.module,
        "path": str(rec.path),
        "lineno": rec.start,
        "end_lineno": rec.end,
        "bases": bases,
        "docstring": doc
    }
    if include_code:
        from .ts_utils import ts_node_text
        e["code"] = ts_node_text(src, node)
    return e


def build_llm_payload(idx: RepoIndex, include_code: bool=False, top_k: int=50, query: Optional[str]=None, depth: int=0, query_weight: float=0.5) -> Dict[str, Any]:
    pr = pagerank_on_calls(idx)
    idf = compute_identifier_idf(idx)

    pr_norm: Dict[str, float] = {}
    if pr:
        m = max(pr.values()) or 1.0
        pr_norm = {k: v / m for k, v in pr.items()}

    base_scores: Dict[str, float] = {}
    for qn, rec in idx.functions.items():
        feats = {
            "loc": rec.node.end_point[0] - rec.node.start_point[0] + 1,
            "cc": rec.node.end_point[0] - rec.node.start_point[0] + 1,  # placeholder, overwritten below
            "pr": pr_norm.get(qn, 0.0),
            "tfidf": function_identifier_tfidf(rec, idf),
            "struct": structure_richness(rec.node),
            "api": api_influence_boost(rec, idx),
        }
        feats["cc"] = cyclomatic_complexity(rec.node)
        feats["loc"] = node_loc(rec.node)
        base_scores[qn] = score_features(feats)

    final_scores = base_scores
    roots: List[str] = []
    if query:
        docs = {qn: f"{idx.functions[qn].name} {qn} {idx.functions[qn].path}" for qn in idx.functions}
        qscore = bm25(query, docs)
        lam = max(0.0, min(1.0, float(query_weight)))
        final_scores = {qn: ((1.0 - lam) * base_scores[qn] + lam * qscore.get(qn, 0.0)) for qn in idx.functions}

    # Enforce top-k on seeds first, then expand and re-cap using fused scores (final_scores)
    seeds, final = select_seeds_and_expand(idx, final_scores, top_k=max(1, top_k), depth=max(0, depth), lexical_min_pct=0.0, query=query)

    selected_funcs: Dict[str, NodeRecord] = {}
    selected_classes: Dict[str, NodeRecord] = {}
    code_blob = None
    if depth > 0 or include_code:
        fn_set = set(final)
        cls_set = set()
        if depth > 0:
            extra_fns, extra_cls = slice_from_roots(final, idx, depth=0)  # classes via our selection logic handled later
            fn_set |= set(extra_fns)
            cls_set |= set(extra_cls)
        selected_funcs = {q: idx.functions[q] for q in fn_set if q in idx.functions}
        selected_classes = {q: idx.classes[q] for q in cls_set if q in idx.classes}
        if include_code:
            code_blob = assemble(selected_funcs, selected_classes)

    # Emit nodes strictly capped and ordered by fused score over the final set
    nodes = [function_entry(idx.functions[q], final_scores.get(q, 0.0), include_code=False) for q in final[:max(1, top_k)]]
    classes = [class_entry(rec, include_code=False) for rec in selected_classes.values()]

    payload: Dict[str, Any] = {
        "summary": {
            "total_functions": len(idx.functions),
            "total_classes": len(idx.classes),
        },
        "nodes": nodes,
        "classes": classes,
        "selected": {
            "roots": seeds,
            "functions": final[:max(1, top_k)],
            "classes": sorted(list(selected_classes.keys())),
        }
    }
    if include_code and code_blob is not None:
        payload["code"] = code_blob
    return payload


