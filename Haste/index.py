#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repository index structures and Python indexing using tree-sitter.
"""

import ast as py_ast
from collections import defaultdict, Counter
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from tree_sitter import Parser as TS_Parser

from .config import STRICT_CC
from .scanner import read_bytes, rel_module_key
from .ts_utils import ts_node_text, walk
from .identifiers import collect_identifiers  # re-export for metrics


class NodeRecord:
    __slots__ = ("path", "module", "lang", "type", "name", "qname", "start", "end", "node", "src_bytes")

    def __init__(self, path: Path, module: str, lang: str, type_: str,
                 name: str, qname: str, start: int, end: int, node, src_bytes: bytes):
        self.path = path
        self.module = module
        self.lang = lang
        self.type = type_
        self.name = name
        self.qname = qname
        self.start = start
        self.end = end
        self.node = node
        self.src_bytes = src_bytes


class RepoIndex:
    def __init__(self, root: Path):
        self.root = root
        self.functions: Dict[str, NodeRecord] = {}   # qname -> record
        self.classes: Dict[str, NodeRecord] = {}     # qname -> record
        self.calls: DefaultDict[str, Counter] = defaultdict(Counter)  # caller_qname -> Counter(callee_symbol)
        self.module_api_info: Dict[Path, Dict[str, Set[str]]] = {}    # path -> {"__all__": set, "main_calls": set}
        # Optional enrichments
        self.variables: Dict[str, NodeRecord] = {}   # qname -> record (module-level variables)
        self.decorators: Dict[str, List[str]] = {}   # qname -> list of decorators as text
        self.wrapper_targets: Dict[str, List[str]] = {}  # function qname -> list of called target texts if wrapper

    def add_function(self, rec: NodeRecord):
        self.functions[rec.qname] = rec

    def add_class(self, rec: NodeRecord):
        self.classes[rec.qname] = rec

    def add_call_edge(self, caller_qname: str, callee_symbol: str):
        if callee_symbol:
            self.calls[caller_qname][callee_symbol] += 1

    def add_variable(self, rec: NodeRecord):
        self.variables[rec.qname] = rec



PY_COMPLEXITY_NODES: Set[str] = {
    "if_statement", "elif_clause",
    "for_statement", "while_statement",
    "except_clause",
    "match_statement",
}


def node_loc(node) -> int:
    return (node.end_point[0] - node.start_point[0] + 1)


def cyclomatic_complexity(node) -> int:
    count = 1
    for n in walk(node):
        if n.type in PY_COMPLEXITY_NODES:
            count += 1
        if STRICT_CC and n.type == "boolean_operator":
            count += 1
    return max(1, count)


def is_public_function_name(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def extract_docstring_if_first_stmt(body_node, src_bytes: bytes) -> Optional[str]:
    if not body_node or body_node.child_count == 0:
        return None
    first = body_node.child(0)
    if first.type == "expression_statement" and first.child_count > 0:
        string_candidate = first.child(0)
        if string_candidate.type in ("string", "concatenated_string"):
            text = ts_node_text(src_bytes, string_candidate)
            try:
                unquoted = py_ast.literal_eval(text)
                return unquoted if isinstance(unquoted, str) else text
            except Exception:
                return text
    return None


def py_function_signature(node, src_bytes: bytes) -> str:
    name_node = node.child_by_field_name("name")
    params_node = node.child_by_field_name("parameters")
    name = ts_node_text(src_bytes, name_node) if name_node else "<anon>"
    params = ts_node_text(src_bytes, params_node) if params_node else "()"
    return f"{name}{params}"


def class_bases_text(node, src_bytes: bytes) -> str:
    header = ts_node_text(src_bytes, node).split(":", 1)[0]
    import re as _re
    m = _re.search(r"\((.*)\)", header, flags=_re.S)
    return (m.group(1).strip() if m else "")




def ast_extract_all_exports(src_text: str) -> Set[str]:
    out: Set[str] = set()
    try:
        m = py_ast.parse(src_text)
    except Exception:
        return out
    for node in m.body:
        if isinstance(node, (py_ast.Assign, py_ast.AnnAssign)):
            targets = []
            if isinstance(node, py_ast.Assign):
                targets = node.targets
            elif isinstance(node, py_ast.AnnAssign) and node.target is not None:
                targets = [node.target]
            if any(isinstance(t, py_ast.Name) and t.id == "__all__" for t in targets):
                val = node.value
                seq = []
                if isinstance(val, (py_ast.List, py_ast.Tuple, py_ast.Set)):
                    seq = val.elts
                if isinstance(val, py_ast.Constant) and isinstance(val.value, (list, tuple, set)):
                    seq = list(val.value)
                for elt in seq:
                    if isinstance(elt, py_ast.Constant) and isinstance(elt.value, str):
                        out.add(elt.value)
    return out


def ast_main_guard_called_names(src_text: str) -> Set[str]:
    calls: Set[str] = set()
    try:
        m = py_ast.parse(src_text)
    except Exception:
        return calls
    for node in m.body:
        if isinstance(node, py_ast.If):
            test = node.test
            ok = False
            if isinstance(test, py_ast.Compare):
                left = test.left
                comps = test.comparators
                if isinstance(left, py_ast.Name) and left.id == "__name__":
                    if comps and isinstance(comps[0], py_ast.Constant) and comps[0].value == "__main__":
                        ok = True
            if ok:
                for sub in py_ast.walk(py_ast.Module(body=node.body, type_ignores=[])):
                    if isinstance(sub, py_ast.Call):
                        fn = sub.func
                        if isinstance(fn, py_ast.Name):
                            calls.add(fn.id)
                        elif isinstance(fn, py_ast.Attribute):
                            calls.add(fn.attr)
    return calls




def function_qualified_name(module_key: str, stack_names: List[str], func_name: str) -> str:
    parts = [module_key] + [s for s in stack_names if s] + [func_name]
    return "::".join(parts)


def index_python_file(p: Path, parser: TS_Parser, idx: RepoIndex):
    src = read_bytes(p)
    tree = parser.parse(src)
    root = tree.root_node

    module_key = rel_module_key(idx.root, p)
    text = src.decode("utf-8", errors="ignore")
    idx.module_api_info[p] = {
        "__all__": ast_extract_all_exports(text),
        "main_calls": ast_main_guard_called_names(text),
    }

    def add_class_node(n, stack: List[str]):
        cname_node = n.child_by_field_name("name")
        cname = ts_node_text(src, cname_node) if cname_node else "<anon>"
        qname = function_qualified_name(module_key, stack, cname)
        idx.add_class(NodeRecord(p, module_key, "python", "class", cname, qname,
                                 n.start_point[0]+1, n.end_point[0]+1, n, src))
        body = n.child_by_field_name("body")
        if body:
            for c in body.children:
                visit(c, stack + [cname])

    def add_function_node(n, stack: List[str]):
        fname_node = n.child_by_field_name("name")
        fname = ts_node_text(src, fname_node) if fname_node else "<anon>"
        qname = function_qualified_name(module_key, stack, fname)
        idx.add_function(NodeRecord(p, module_key, "python", "function", fname, qname,
                                    n.start_point[0]+1, n.end_point[0]+1, n, src))
        # Wrapper heuristic: single call or return call in body (ignoring docstring)
        def _is_wrapper(fn_node) -> Tuple[bool, List[str]]:
            body = fn_node.child_by_field_name("body")
            if body is None:
                return False, []
            stmts = [c for c in body.children if c.type not in (":",)]
            # skip docstring expression
            if stmts and stmts[0].type == "expression_statement" and stmts[0].child_count > 0 and stmts[0].child(0).type in ("string", "concatenated_string"):
                stmts = stmts[1:]
            if len(stmts) != 1:
                return False, []
            s = stmts[0]
            targets: List[str] = []
            if s.type == "expression_statement" and s.child_count > 0:
                expr = s.child(0)
                if expr.type == "call" and expr.child_count > 0:
                    targets.append(ts_node_text(src, expr.child_by_field_name("function") or expr.child(0)))
            if s.type == "return_statement" and s.child_count > 0:
                expr = s.child(s.child_count - 1)
                if expr.type == "call" and expr.child_count > 0:
                    targets.append(ts_node_text(src, expr.child_by_field_name("function") or expr.child(0)))
            return (len(targets) > 0), targets

        is_wrap, wrap_targets = _is_wrapper(n)
        if is_wrap:
            # store raw text; resolve later if needed
            idx.wrapper_targets[qname] = wrap_targets
        for sub in walk(n):
            if sub.type == "call":
                target = sub.child_by_field_name("function")
                callee_name = ""
                if target:
                    if target.type == "identifier":
                        callee_name = ts_node_text(src, target)
                    elif target.type == "attribute":
                        attr = target.child_by_field_name("attribute")
                        if attr and attr.type == "identifier":
                            callee_name = ts_node_text(src, attr)
                if callee_name:
                    idx.add_call_edge(qname, callee_name)

        body = n.child_by_field_name("body")
        if body:
            for c in body.children:
                visit(c, stack + [fname])

    def visit(n, stack: List[str]):
        t = n.type
        if t == "decorated_definition":
            # collect decorators text
            decorators: List[str] = []
            target_node = None
            for c in n.children:
                if c.type == "decorator":
                    deco_txt = ts_node_text(src, c).lstrip("@").strip()
                    if deco_txt:
                        decorators.append(deco_txt)
                if c.type in ("function_definition", "class_definition", "async_function_definition"):
                    target_node = c
            if target_node is not None:
                # compute qname to record decorators
                name_node = target_node.child_by_field_name("name")
                name = ts_node_text(src, name_node) if name_node else ""
                qname = function_qualified_name(module_key, stack, name)
                if decorators:
                    idx.decorators[qname] = decorators
                visit(target_node, stack)
            return
        if t in ("function_definition", "async_function_definition"):
            add_function_node(n, stack)
            return
        if t == "class_definition":
            add_class_node(n, stack)
            return
        for c in n.children:
            visit(c, stack)

    # Collect top-level variables (simple assignments)
    for top in root.children:
        if top.type in ("assignment", "augmented_assignment", "type_alias_statement", "annotated_assignment"):
            # extract left-hand names heuristically
            head = ts_node_text(src, top).split("=", 1)[0]
            for piece in head.split(","):
                name = piece.strip()
                if name and "." not in name and not name.startswith("(") and not name.startswith("[") and name.replace("_", "").isalnum():
                    qn = function_qualified_name(module_key, [], name)
                    idx.add_variable(NodeRecord(p, module_key, "python", "variable", name, qn,
                                                top.start_point[0]+1, top.end_point[0]+1, top, src))

    visit(root, [])


