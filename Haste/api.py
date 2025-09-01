#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Public, ergonomic API for haste.

Facade exposing a minimal, stable interface so users can:
- select from a single file with one call (mirrors CLI behavior)
- build a repo-level payload with one call
- extract structural context from source code
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cast_chunker import ByteSpan as Span
from .cast_chunker import cast_split_merge
from .exporter import stitch_code
from .index import RepoIndex
from .index import index_python_file as ts_index_python_file
from .index_py import Symbol, index_python_file
from .payload import build_llm_payload
from .retriever import (Doc, bfs_expand, build_bm25_corpus, lexical_topk,
                        semantic_rerank)
# Repo-level imports
from .scanner import iter_source_files
from .ts_utils import build_ts_parser, ts_node_text, walk


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


def select_from_file(
    path: str,
    query: str,
    *,
    top_k: int = 6,
    prefilter: int = 300,
    bfs_depth: int = 1,
    max_add: int = 12,
    semantic: bool = False,
    sem_model: str = "text-embedding-3-small",
    hard_cap: int = 1200,
    soft_cap: int = 1800,
) -> Dict[str, Any]:
    if hard_cap <= 0 or soft_cap <= 0:
        raise ValueError("hard_cap and soft_cap must be positive integers")
    if soft_cap < hard_cap:
        soft_cap = hard_cap

    src_bytes, symbols, _aliases = index_python_file(path)
    docs = _to_doc_list(symbols)
    bm25, _ = build_bm25_corpus(docs)

    prelim = lexical_topk(docs, bm25, query, k=top_k, prefilter=prefilter)
    if semantic:
        prelim = semantic_rerank(prelim, query, sem_model, src_bytes=src_bytes)
    if not prelim:
        prelim = lexical_topk(docs, bm25, query, k=top_k, prefilter=max(30, top_k))

    call_edges = _build_call_edges(symbols)
    docs_by_name: Dict[str, List[Doc]] = {}
    for d in docs:
        docs_by_name.setdefault(d.name, []).append(d)
    expanded = bfs_expand(prelim[: top_k], docs_by_name, call_edges, depth=bfs_depth, max_add=max_add)

    spans = [Span(d.start_byte, d.end_byte) for d in expanded]
    stitched_spans = cast_split_merge(src_bytes, spans, hard_cap_tokens=hard_cap, soft_cap_tokens=soft_cap)
    code, _mapping = stitch_code(src_bytes, stitched_spans)

    # Nodes payload with line numbers
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
    return out


def build_payload_from_repo(
    root: str | Path,
    *,
    include_code: bool = False,
    top_k: int = 50,
    depth: int = 0,
    query: Optional[str] = None,
    query_weight: float = 0.5,
) -> Dict[str, Any]:
    root_path = Path(root)
    parser = build_ts_parser("python")
    idx = RepoIndex(root_path)
    for p in iter_source_files(root_path):
        ts_index_python_file(p, parser, idx)
    return build_llm_payload(
        idx,
        include_code=include_code,
        top_k=top_k,
        query=query,
        depth=depth,
        query_weight=query_weight,
    )


def build_structural_context_from_source(source_text: str, lang_name: str = "python") -> str:
    """Return a concise, relevant structural summary using tree-sitter.

    Includes module docstring (truncated), imports, classes with bases, and
    top-level functions and class methods with signatures.
    """
    try:
        parser = build_ts_parser(lang_name)
    except Exception as e:
        return f"[AST unavailable: {e}]"

    source_bytes = bytes(source_text, "utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    lines = []

    # Module docstring (best-effort): first statement is a string
    for child in root.children:
        if child.type == "expression_statement" and child.child_count > 0 and child.children[0].type == "string":
            doc = ts_node_text(source_bytes, child.children[0])
            doc = doc.strip().strip('"').strip("'")
            if doc:
                if len(doc) > 200:
                    doc = doc[:200] + "…"
                lines.append(f"Docstring: {doc}")
            break

    # Imports
    imports = []
    for node in walk(root):
        if node.type in ("import_statement", "import_from_statement"):
            imports.append(ts_node_text(source_bytes, node).strip())
    if imports:
        lines.append("Imports:")
        lines.extend(f"  {imp}" for imp in imports[:50])

    # Classes and functions
    class_entries = []
    function_entries = []
    class_methods = {}

    def get_identifier(n):
        for c in n.children:
            if c.type == "identifier":
                return ts_node_text(source_bytes, c)
        return None

    def get_child_text(n, type_name):
        for c in n.children:
            if c.type == type_name:
                return ts_node_text(source_bytes, c)
        return None

    for node in walk(root):
        if node.type == "function_definition":
            name = get_identifier(node) or "<anon>"
            params = get_child_text(node, "parameters") or "()"
            sig = f"def {name}{params}"
            # Determine if nested in a class
            parent = node.parent
            while parent is not None and parent.type not in ("class_definition", "module"):
                parent = parent.parent
            if parent is not None and parent.type == "class_definition":
                cls_name = get_identifier(parent) or "<anon>"
                class_methods.setdefault(cls_name, []).append(sig)
            else:
                function_entries.append(sig)
        elif node.type == "class_definition":
            name = get_identifier(node) or "<anon>"
            bases = get_child_text(node, "argument_list")  # optional
            entry = f"class {name}{bases if bases else ''}"
            class_entries.append(entry)

    if class_entries:
        lines.append("Classes:")
        for entry in class_entries[:200]:
            lines.append(f"  {entry}")
            cls_name = entry.split()[1].split("(")[0]
            methods = class_methods.get(cls_name, [])
            for m in methods[:200]:
                lines.append(f"    {m}")

    if function_entries:
        lines.append("Functions:")
        lines.extend(f"  {fn}" for fn in function_entries[:300])

    if not lines:
        return "[No significant structural elements detected]"
    return "\n".join(lines)


