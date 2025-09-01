# SPDX-License-Identifier: MIT
# Lightweight Python indexer with Tree-sitter and a tiny symbol table.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import os
from tree_sitter import Parser, Node  # type: ignore

# Use the latest tree-sitter-language-pack which returns a proper tree_sitter.Language
try:
    from tree_sitter_language_pack import get_language  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Install 'tree-sitter-language-pack' for language support: pip install tree-sitter-language-pack"
    ) from e

PY_LANGUAGE = get_language("python")


# --- Tree-sitter setup ---
parser = Parser(PY_LANGUAGE)


@dataclass
class Symbol:
    qname: str                 # qualified name like module::Class.method or module::func
    kind: str                  # "function" | "class" | "variable"
    name: str
    module: str
    path: str
    start_byte: int
    end_byte: int
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    docstring: str = ""
    identifiers: List[str] = field(default_factory=list)  # split identifiers in body
    calls: List[str] = field(default_factory=list)        # callee unqualified names (resolved best effort)
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)   # decorator list like ["staticmethod", "wraps(x)"]
    is_wrapper: bool = False                              # simple pass-through wrapper
    wrapper_targets: List[str] = field(default_factory=list)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _node_text(src: bytes, n: Node) -> bytes:
    return src[n.start_byte:n.end_byte]


def _maybe_docstring(src: bytes, body_node: Node) -> str:
    # Pull first string literal in a suite as docstring
    for ch in body_node.children:
        if ch.type == "expression_statement" and len(ch.children) and ch.children[0].type == "string":
            raw = _node_text(src, ch.children[0]).decode("utf-8", errors="replace")
            return raw.strip(' \n\r"\'')
    return ""


def _gather_identifiers(src: bytes, node: Node, out: List[str]) -> None:
    # harvest identifiers and dotted names for lexical matching
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in ("identifier", "attribute"):
            out.append(_node_text(src, n).decode("utf-8", errors="ignore"))
        stack.extend(n.children)


def _gather_calls(src: bytes, node: Node) -> List[str]:
    calls: List[str] = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "call":
            # function being called is first child (could be identifier or attribute)
            if n.children:
                tgt = n.children[0]
                calls.append(_node_text(src, tgt).decode("utf-8", errors="ignore"))
        stack.extend(n.children)
    return calls


def _collect_decorators(src: bytes, decorated: Node) -> Tuple[List[str], Optional[Node]]:
    """Return (decorators_text, underlying_def_node) for a decorated_definition."""
    decos: List[str] = []
    target_def: Optional[Node] = None
    for ch in decorated.children:
        if ch.type == "decorator":
            # '@' + expression; we want the expression text
            # children often: '@', dotted_name/call, NEWLINE
            # Take all children except leading '@'
            piece = _node_text(src, ch).decode("utf-8", "ignore").lstrip("@").strip()
            if piece:
                decos.append(piece)
        if ch.type in ("function_definition", "class_definition", "async_function_definition"):
            target_def = ch
    return decos, target_def


def _is_wrapper_function(src: bytes, suite: Optional[Node]) -> Tuple[bool, List[str]]:
    """Heuristic: function body has a single call or return of a call."""
    if suite is None:
        return False, []
    # find statements inside suite/block
    stmts = [c for c in suite.children if c.type not in (":",)]
    # filter out docstring expr if present
    if stmts and stmts[0].type == "expression_statement" and len(stmts[0].children) and stmts[0].children[0].type == "string":
        stmts = stmts[1:]
    if len(stmts) != 1:
        return False, []
    s = stmts[0]
    targets: List[str] = []
    if s.type == "expression_statement" and len(s.children):
        expr = s.children[0]
        if expr.type == "call":
            tgt = expr.children[0] if expr.children else None
            if tgt is not None:
                targets.append(_node_text(src, tgt).decode("utf-8", "ignore"))
    if s.type == "return_statement" and len(s.children):
        expr = s.children[-1]
        if expr.type == "call":
            tgt = expr.children[0] if expr.children else None
            if tgt is not None:
                targets.append(_node_text(src, tgt).decode("utf-8", "ignore"))
    return (len(targets) > 0), targets


def _collect_module_variables(src: bytes, root: Node) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    """Collect simple top-level assignments as variable names.
    Returns list of (name, start_point, end_point).
    """
    vars_out: List[Tuple[str, Tuple[int, int], Tuple[int, int]]] = []
    for ch in root.children:
        if ch.type in ("assignment", "augmented_assignment", "type_alias_statement", "annotated_assignment"):
            # Heuristic: take text up to '=' and split by ','; pick identifierish names
            txt = _node_text(src, ch).decode("utf-8", "ignore")
            head = txt.split("=", 1)[0]
            for cand in head.split(","):
                name = cand.strip()
                if name and name.replace("_", "").replace(".", "").isalnum() and " " not in name and "(" not in name:
                    # Exclude attributes like self.x; keep bare identifiers only
                    if "." in name:
                        continue
                    vars_out.append((name, (ch.start_point.row, ch.start_point.column), (ch.end_point.row, ch.end_point.column)))
    return vars_out


def _symbol_signature(src: bytes, def_node: Node) -> str:
    # cheap signature for function defs
    name = ""
    params = ""
    for ch in def_node.children:
        if ch.type == "identifier":
            name = _node_text(src, ch).decode("utf-8", "ignore")
        if ch.type == "parameters":
            params = _node_text(src, ch).decode("utf-8", "ignore")
    return f"{name}{params}"


def _module_name(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".py"):
        return base[:-3]
    return base


def index_python_file(path: str) -> tuple[bytes, List[Symbol], Dict[str, str]]:
    """
    Returns (source_bytes, symbols, import_aliases)
    import_aliases: local name -> fully-qualified (best effort)
    """
    src = _read_bytes(path)
    tree = parser.parse(src)
    root = tree.root_node
    module = _module_name(path)
    symbols: List[Symbol] = []
    aliases: Dict[str, str] = {}

    # collect import aliases: "import x as y" / "from a.b import c as d"
    def visit_imports(n: Node) -> None:
        if n.type == "import_statement":
            names: List[str] = []
            for ch in n.children:
                if ch.type in ("dotted_name", "aliased_import"):
                    names.append(_node_text(src, ch).decode("utf-8", "ignore"))
            for item in names:
                if " as " in item:
                    full, local = [x.strip() for x in item.split(" as ", 1)]
                    aliases[local] = full
                else:
                    head = item.split(".", 1)[0]
                    aliases[head] = item
        if n.type == "import_from_statement":
            txt = _node_text(src, n).decode("utf-8", "ignore")
            try:
                head = txt.split("from", 1)[1].split("import", 1)[0].strip()
                tail = txt.split("import", 1)[1]
                for piece in tail.split(","):
                    piece = piece.strip()
                    if not piece:
                        continue
                    if " as " in piece:
                        full, local = [x.strip() for x in piece.split(" as ", 1)]
                        aliases[local] = f"{head}.{full}"
                    else:
                        aliases[piece] = f"{head}.{piece}"
            except Exception:
                pass

    # Collect module-level variables as symbols
    for (vname, sp, ep) in _collect_module_variables(src, root):
        symbols.append(Symbol(
            qname=f"{module}::{vname}",
            kind="variable",
            name=vname,
            module=module,
            path=path,
            start_byte=0,
            end_byte=0,
            start_point=sp,
            end_point=ep,
        ))

    # Walk top-level
    for ch in root.children:
        if ch.type in ("import_statement", "import_from_statement"):
            visit_imports(ch)
        target_node: Optional[Node] = None
        decorators: List[str] = []
        if ch.type == "decorated_definition":
            decorators, target_node = _collect_decorators(src, ch)
        if ch.type in ("class_definition", "function_definition"):
            target_node = ch
        if target_node is not None:
            kind = "class" if ch.type == "class_definition" else "function"
            if ch.type == "decorated_definition":
                # recompute kind using target_node
                kind = "class" if target_node.type == "class_definition" else "function"
            name = ""
            suite: Optional[Node] = None
            for c2 in target_node.children:
                if c2.type == "identifier":
                    name = _node_text(src, c2).decode("utf-8", "ignore")
                if c2.type in ("block", "suite"):
                    suite = c2
            qname = f"{module}::{name}"
            doc = _maybe_docstring(src, suite) if suite else ""
            ids: List[str] = []
            if suite:
                _gather_identifiers(src, suite, ids)
            calls = _gather_calls(src, suite) if suite else []
            signature = _symbol_signature(src, target_node)
            is_wrapper, wrapper_targets = _is_wrapper_function(src, suite) if kind == "function" else (False, [])

            symbols.append(Symbol(
                qname=qname,
                kind=kind,
                name=name,
                module=module,
                path=path,
                start_byte=target_node.start_byte,
                end_byte=target_node.end_byte,
                start_point=(target_node.start_point.row, target_node.start_point.column),
                end_point=(target_node.end_point.row, target_node.end_point.column),
                docstring=doc,
                identifiers=ids,
                calls=calls,
                signature=signature,
                decorators=decorators,
                is_wrapper=is_wrapper,
                wrapper_targets=wrapper_targets,
            ))

    return src, symbols, aliases


