#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration constants for haste (legacy: ast_compress).

Holds language mappings, ignore lists, and feature flags used across the
package. Keep this module dependency-free to avoid import cycles.
"""

from typing import Dict, Set


# Map file extensions to language names recognized by tree-sitter-language-pack
LANG_BY_EXT: Dict[str, str] = {
    ".py": "python",
}


# Directory names to skip when scanning a repository
IGNORE_DIRS: Set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "site-packages",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
}


# Feature flags
STRICT_CC: bool = True  # Count boolean operators as extra decisions in CC


