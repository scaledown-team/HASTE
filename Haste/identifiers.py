#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Identifier extraction and filtering for TF-IDF.
"""

from typing import List, Set
import ast as py_ast  # only for literal_eval safety if needed elsewhere
import keyword
import builtins

from .ts_utils import ts_node_text, walk


PY_STOP_TERMS: Set[str] = set(
    t.lower() for t in (
        list(keyword.kwlist) +
        dir(builtins) +
        [
            "self", "cls", "args", "kwargs", "true", "false", "none",
            "data", "value", "values", "result", "item", "items",
            "key", "keys", "val", "vals", "tmp", "obj", "objs",
            "out", "ret", "return", "input", "output",
        ]
    )
)


def is_informative_identifier(tok: str) -> bool:
    if not tok:
        return False
    t = tok.lower()
    if t in PY_STOP_TERMS:
        return False
    if t.startswith("__") and t.endswith("__"):
        return False
    if set(t) == {"_"}:
        return False
    if len(t) < 2:
        return False
    return True


def collect_identifiers(node, src_bytes: bytes, filtered: bool = True) -> List[str]:
    ids: List[str] = []
    for n in walk(node):
        if n.type == "identifier":
            tok = ts_node_text(src_bytes, n)
            ids.append(tok)
        if n.type == "attribute":
            attr = n.child_by_field_name("attribute")
            if attr and attr.type == "identifier":
                ids.append(ts_node_text(src_bytes, attr))
    if filtered:
        ids = [t for t in ids if is_informative_identifier(t)]
    return ids



