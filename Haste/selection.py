#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Selection, slicing, and assembly of code snippets for LLM consumption.
"""

import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .index import RepoIndex, NodeRecord


def _symbol_index(functions: Dict[str, NodeRecord]) -> Dict[str, Set[str]]:
    sym2q: Dict[str, Set[str]] = {}
    for q, rec in functions.items():
        sym2q.setdefault(rec.name, set()).add(q)
    return sym2q


def slice_from_roots(roots: List[str], idx: RepoIndex, depth: int = 1) -> Tuple[Set[str], Set[str]]:
    sym2q = _symbol_index(idx.functions)
    selected: Set[str] = set()
    queue: List[Tuple[str, int]] = [(r, 0) for r in roots if r in idx.functions]
    while queue:
        u, d = queue.pop(0)
        if u in selected:
            continue
        selected.add(u)
        if d >= depth:
            continue
        for sym, _cnt in idx.calls.get(u, {}).items():
            for v in sym2q.get(sym, ()):  # may match multiple qnames
                if v not in selected:
                    queue.append((v, d + 1))

    needed_classes: Set[str] = set()
    by_module_classes: Dict[str, List[Tuple[str, NodeRecord]]] = {}
    for cq, crec in idx.classes.items():
        by_module_classes.setdefault(crec.module, []).append((cq, crec))
    for q in selected:
        rec = idx.functions[q]
        code = rec.src_bytes[rec.node.start_byte:rec.node.end_byte].decode("utf-8", "ignore")
        for cq, crec in by_module_classes.get(rec.module, []):
            if re.search(rf"\b{re.escape(crec.name)}\b", code):
                needed_classes.add(cq)

    # --- Annotation-aware inclusion (type closure) ---
    # Map (module, class_name) -> qname
    cls_index: Dict[Tuple[str, str], str] = {}
    for q, crec in idx.classes.items():
        cls_index[(crec.module, crec.name)] = q

    # Collect annotation type names within a source line region
    def _names_from_annotation(node: ast.AST) -> Set[str]:
        out: Set[str] = set()
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
            out |= _names_from_annotation(node.value)
        elif isinstance(node, ast.Subscript):
            out |= _names_from_annotation(node.value)
            sl = node.slice
            if isinstance(sl, ast.Tuple):
                for e in sl.elts:
                    out |= _names_from_annotation(e)
            else:
                out |= _names_from_annotation(sl)
        elif isinstance(node, ast.Call):
            out |= _names_from_annotation(node.func)
            for a in node.args:
                out |= _names_from_annotation(a)
            for kw in node.keywords:
                if kw.value is not None:
                    out |= _names_from_annotation(kw.value)
        elif isinstance(node, ast.BinOp):
            out |= _names_from_annotation(node.left)
            out |= _names_from_annotation(node.right)
        elif isinstance(node, ast.UnaryOp):
            out |= _names_from_annotation(node.operand)
        return out

    def _annotation_types_in_region(py_path: str, start: int, end: int) -> Set[str]:
        try:
            text = Path(py_path).read_text(encoding="utf-8", errors="ignore")
            mod = ast.parse(text)
        except Exception:
            return set()
        want: Set[str] = set()
        for n in ast.walk(mod):
            ln = getattr(n, "lineno", None)
            en = getattr(n, "end_lineno", ln)
            if ln is None:
                continue
            if ln < start or (en is not None and en > end):
                continue
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in list(n.args.args) + list(n.args.kwonlyargs):
                    if arg.annotation:
                        want |= _names_from_annotation(arg.annotation)
                if n.args.vararg and n.args.vararg.annotation:
                    want |= _names_from_annotation(n.args.vararg.annotation)
                if n.args.kwarg and n.args.kwarg.annotation:
                    want |= _names_from_annotation(n.args.kwarg.annotation)
                if n.returns:
                    want |= _names_from_annotation(n.returns)
            elif isinstance(n, ast.AnnAssign) and n.annotation:
                want |= _names_from_annotation(n.annotation)
        return want

    # 1) From selected functions
    for q in list(selected):
        rec = idx.functions[q]
        if rec.lang != "python":
            continue
        names = _annotation_types_in_region(str(rec.path), rec.start, rec.end)
        for nm in names:
            qmatch = cls_index.get((rec.module, nm))
            if qmatch:
                needed_classes.add(qmatch)

    # 2) From already selected classes (their fields may reference types)
    for q in list(needed_classes):
        crec = idx.classes[q]
        names = _annotation_types_in_region(str(crec.path), crec.start, crec.end)
        for nm in names:
            qmatch = cls_index.get((crec.module, nm))
            if qmatch:
                needed_classes.add(qmatch)
    return selected, needed_classes


def assemble(selected_funcs: Dict[str, NodeRecord], selected_classes: Dict[str, NodeRecord]) -> str:
    files = sorted(set([r.path for r in selected_funcs.values()] + [r.path for r in selected_classes.values()]))
    imports_by_file: Dict[str, List[str]] = {}

    def _top_level_imports(py_path: str) -> List[str]:
        out: List[str] = []
        try:
            text = Path(py_path).read_text(encoding="utf-8", errors="ignore")
            mod = ast.parse(text)
        except Exception:
            return out
        for n in mod.body:
            if isinstance(n, ast.Import):
                names = ", ".join([a.name if not a.asname else f"{a.name} as {a.asname}" for a in n.names])
                out.append(f"import {names}")
            elif isinstance(n, ast.ImportFrom):
                modname = n.module or ""
                names = ", ".join([a.name if not a.asname else f"{a.name} as {a.asname}" for a in n.names])
                level = "." * n.level if getattr(n, "level", 0) else ""
                out.append(f"from {level}{modname} import {names}")
        return out

    for path in files:
        imports_by_file[str(path)] = _top_level_imports(str(path))

    def snippet(rec: NodeRecord) -> str:
        return rec.src_bytes[rec.node.start_byte:rec.node.end_byte].decode("utf-8", "ignore")

    out_parts: List[str] = []
    for path in files:
        p = str(path)
        if imports_by_file.get(p):
            out_parts.extend(imports_by_file[p]); out_parts.append("")
        cls_here = [c for c in selected_classes.values() if str(c.path) == p]
        fn_here  = [f for f in selected_funcs.values()  if str(f.path) == p]
        if cls_here:
            out_parts.append("\n\n".join(snippet(c) for c in cls_here)); out_parts.append("")
        if fn_here:
            out_parts.append("\n\n".join(snippet(f) for f in fn_here)); out_parts.append("")
    return "\n".join(out_parts).strip()


