#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filesystem scanning helpers: iterate source files and compute stable module keys.
"""

import os
from pathlib import Path
from typing import Iterable

from .config import LANG_BY_EXT, IGNORE_DIRS


def should_skip_dir(dirname: str) -> bool:
    name = dirname.strip()
    if name in IGNORE_DIRS:
        return True
    if name.startswith(".") and name not in {".", ".."}:
        return True
    return False


def iter_source_files(root: Path) -> Iterable[Path]:
    root = root.resolve()
    if root.is_file():
        if root.suffix in LANG_BY_EXT:
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fn in filenames:
            p = Path(dirpath, fn)
            if p.suffix in LANG_BY_EXT:
                yield p


def rel_module_key(root: Path, p: Path) -> str:
    """Return a stable module key (relative path with `/`).

    Works when root is a file or a directory; falls back to filename on failure.
    """
    root = root.resolve()
    p = p.resolve()
    base = root.parent if root.is_file() else root
    try:
        rel = p.relative_to(base)
        rel_str = str(rel).replace(os.sep, "/")
        return rel_str if rel_str and rel_str != "." else p.name
    except Exception:
        return p.name


def read_bytes(p: Path) -> bytes:
    return p.read_bytes()



