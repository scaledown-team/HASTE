#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line interface for CAST chunking + hybrid retrieval over a single Python file.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List, Dict

from .index_py import index_python_file, Symbol
from .cast_chunker import cast_split_merge, ByteSpan as Span
from .exporter import stitch_code
from .retriever import Doc, build_bm25_corpus, lexical_topk, semantic_rerank, bfs_expand
import bisect


def _to_doc_list(symbols: List[Symbol]) -> List[Doc]:
    docs: List[Doc] = []
    for i, s in enumerate(symbols):
        docs.append(Doc(
            idx=i,
            module=s.module,
            qname=s.qname,
            kind=s.kind,
            name=s.name,
            path=s.path,
            docstring=s.docstring,
            identifiers=s.identifiers,
            signature=s.signature or "",
            start_byte=s.start_byte,
            end_byte=s.end_byte,
        ))
    return docs


def _build_call_edges(symbols: List[Symbol]) -> Dict[str, List[str]]:
    edges: Dict[str, List[str]] = {}
    for s in symbols:
        out: List[str] = []
        for c in s.calls:
            base = c.split(".")[-1]
            if base not in out:
                out.append(base)
        edges[s.qname] = out
    return edges


def _build_line_starts(src_bytes: bytes) -> list[int]:
    starts = [0]
    find = src_bytes.find
    i = 0
    while True:
        j = find(b"\n", i)
        if j == -1:
            break
        starts.append(j + 1)
        i = j + 1
    return starts


def _byte_to_line(byte_off: int, line_starts: list[int]) -> int:
    return bisect.bisect_right(line_starts, byte_off)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to a .py file (Python only in this minimal CLI).")
    ap.add_argument("--query", required=True, help="Free-form text query.")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--prefilter", type=int, default=300)
    ap.add_argument("--bfs-depth", type=int, default=1)
    ap.add_argument("--max-add", type=int, default=12)
    ap.add_argument("--semantic", action="store_true", default=False)
    ap.add_argument("--sem-model", default="text-embedding-3-small")
    ap.add_argument("--hard-cap", type=int, default=1200)
    ap.add_argument("--soft-cap", type=int, default=1800)
    args = ap.parse_args()

    if args.hard_cap <= 0 or args.soft_cap <= 0:
        print(json.dumps({"error": "hard/soft caps must be positive integers"}))
        return 2
    if args.soft_cap < args.hard_cap:
        # keep it safe instead of crashing; soften by raising soft cap
        args.soft_cap = args.hard_cap
    if not os.path.isfile(args.path) or not args.path.endswith(".py"):
        print(json.dumps({"error": "Only single Python file path is supported in this minimal CLI."}))
        return 2

    # Index file
    src_bytes, symbols, _aliases = index_python_file(args.path)
    docs = _to_doc_list(symbols)
    bm25, _ = build_bm25_corpus(docs)

    prelim = lexical_topk(docs, bm25, args.query, k=args.top_k, prefilter=args.prefilter)
    if args.semantic:
        prelim = semantic_rerank(prelim, args.query, args.sem_model, src_bytes=src_bytes)

    # If still empty (e.g., odd query), degrade to top lexical few
    if not prelim:
        prelim = lexical_topk(docs, bm25, args.query, k=args.top_k, prefilter=max(30, args.top_k))

    # BFS expand on call edges
    call_edges = _build_call_edges(symbols)
    docs_by_name: Dict[str, List[Doc]] = {}
    for d in docs:
        docs_by_name.setdefault(d.name, []).append(d)

    expanded = bfs_expand(prelim[: args.top_k], docs_by_name, call_edges, depth=args.bfs_depth, max_add=args.max_add)

    # CAST over selected spans
    spans = [Span(d.start_byte, d.end_byte) for d in expanded]
    stitched_spans = cast_split_merge(src_bytes, spans, hard_cap_tokens=args.hard_cap, soft_cap_tokens=args.soft_cap)
    code, _mapping = stitch_code(src_bytes, stitched_spans)

    # Prepare JSON
    nodes = []
    line_starts = _build_line_starts(src_bytes)
    for d in expanded:
        lineno = _byte_to_line(d.start_byte, line_starts)
        end_lineno = _byte_to_line(max(d.end_byte - 1, 0), line_starts)
        nodes.append({
            "type": d.kind,
            "name": d.name,
            "qname": d.qname,
            "module": d.module,
            "path": d.path,
            "lineno": lineno,
            "end_lineno": end_lineno,
            "signature": d.signature,
            "docstring": d.docstring or None,
            "score": d.score,
        })

    out = {
            "summary": {
            "total_functions": sum(1 for s in symbols if s.kind == "function"),
            "total_classes": sum(1 for s in symbols if s.kind == "class"),
            },
        "nodes": nodes,
        "classes": [n for n in nodes if n["type"] == "class"],
            "selected": {
            "roots": [d.qname for d in expanded],
            "functions": [d.qname for d in expanded if d.kind == "function"],
            "classes": [d.qname for d in expanded if d.kind == "class"],
        },
        "code": code,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


