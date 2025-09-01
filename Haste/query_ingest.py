import os
import re
import json
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple


_PYTEST_NODEID = re.compile(r"(?P<path>[^:\s]+)(::(?P<class>[\w\.]+))?(::(?P<func>[\w\.]+))?")
_STACK_LINE = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')
_DIFF_FILE = re.compile(r'^\+\+\+ b/(.+)$|^--- a/(.+)$')
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_PIECE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")


def _split_identifiers(text: str) -> List[str]:
    toks: List[str] = []
    for tok in _IDENT.findall(text or ""):
        toks.append(tok)
        toks.extend([p.lower() for p in _CAMEL_PIECE.findall(tok)])
        toks.extend([p for p in tok.split("_") if p])
    out: List[str] = []
    for t in toks:
        for suf in ("ing", "ers", "er", "ed", "ies", "s"):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                t = t[: -len(suf)]
                break
        out.append(t.lower())
    # unique + sorted for stability
    seen: Set[str] = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


@dataclass
class QuerySignals:
    raw: str
    keywords: Set[str]
    files: Set[str]
    nodeids: Set[str]
    functions: Set[str]
    classes: Set[str]
    stack_hits: List[Tuple[str, int, str]]
    diffs: Set[str]


def _read_text_maybe(path_or_text: str) -> str:
    if path_or_text == "-" and not sys.stdin.isatty():
        return sys.stdin.read()
    if os.path.exists(path_or_text) and os.path.isfile(path_or_text):
        try:
            return open(path_or_text, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            pass
    return path_or_text


def ingest_query(anything: str) -> QuerySignals:
    raw = _read_text_maybe(anything)
    try:
        js = json.loads(raw)
        candidates: List[str] = []
        for k in ("title", "body", "message", "error", "trace", "description", "hint", "prompt", "query", "task"):
            v = js.get(k)
            if isinstance(v, str):
                candidates.append(v)
        for k in ("tests", "test_list", "hints", "files", "patch", "diff"):
            v = js.get(k)
            if isinstance(v, list):
                candidates.extend([str(x) for x in v])
            elif isinstance(v, str):
                candidates.append(v)
        if candidates:
            raw = "\n".join(candidates)
    except Exception:
        pass

    files: Set[str] = set()
    nodeids: Set[str] = set()
    functions: Set[str] = set()
    classes: Set[str] = set()
    diffs: Set[str] = set()
    stack_hits: List[Tuple[str, int, str]] = []

    for line in (raw or "").splitlines():
        m = _PYTEST_NODEID.search(line)
        if m and m.group("path"):
            nodeids.add(m.group(0))
            files.add(m.group("path"))

        sm = _STACK_LINE.search(line)
        if sm:
            try:
                stack_hits.append((sm.group(1), int(sm.group(2)), sm.group(3)))
            except Exception:
                pass
            files.add(sm.group(1))

        dm = _DIFF_FILE.match(line)
        if dm:
            path = dm.group(1) or dm.group(2)
            if path:
                diffs.add(path)
                files.add(path)

        for ident in _IDENT.findall(line):
            if ident and ident[:1].isupper():
                classes.add(ident)
            else:
                functions.add(ident)

        if ("/" in line or "\\" in line) and (line.endswith(".py") or ".py:" in line):
            p = line.split()[0]
            files.add(p.split(":")[0])

    keywords = set(_split_identifiers(raw))
    return QuerySignals(
        raw=raw,
        keywords=keywords,
        files=files,
        nodeids=nodeids,
        functions=functions,
        classes=classes,
        stack_hits=stack_hits,
        diffs=diffs,
    )



