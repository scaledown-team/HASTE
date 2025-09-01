#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-sitter utilities.

Provides parser construction, generic node traversal, and safe node text
extraction across the package.
"""

from typing import Iterable

from tree_sitter import Parser as TS_Parser
try:
    # Maintained bundle matching tree-sitter>=0.23/0.25 APIs
    from tree_sitter_language_pack import get_parser  # type: ignore
except ModuleNotFoundError as _e:  # pragma: no cover - fail loudly at runtime
    raise RuntimeError(
        "tree-sitter-language-pack is required with modern tree-sitter. "
        "Install it (e.g., `poetry add tree-sitter-language-pack`) and rerun."
    ) from _e


def build_ts_parser(lang_name: str) -> TS_Parser:
    """Return a configured tree-sitter Parser bound to the language name."""
    return get_parser(lang_name)


def ts_node_text(source: bytes, node) -> str:
    """Safely decode the bytes covering the node span to UTF-8 text."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def walk(node) -> Iterable:
    """Iterative DFS over a tree-sitter node's descendants (node included)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))



