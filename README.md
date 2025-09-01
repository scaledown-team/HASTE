## HasteContext

Parser-backed code-context compression for Python using Tree-sitter. It builds a structured index of functions/classes, ranks relevant functions for a free‑form query with lexical BM25 (optionally fused with semantic embeddings), expands along the call graph, then assembles a compact, LLM-ready payload. A minimal CLI is included for single-file workflows; the library API supports repository-level indexing and hybrid selection.

[![PyPI version](https://badge.fury.io/py/HasteContext.svg)](https://badge.fury.io/py/HasteContext)
[![Python Versions](https://img.shields.io/pypi/pyversions/HasteContext.svg)](https://pypi.org/project/HasteContext/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🔗 [PyPI Project Page](https://pypi.org/project/HasteContext/0.2.1/)

Import name is `haste` for API compatibility.

### What's New in 0.2.1
- Fixed badge version information
- Improved package structure with better encapsulation of implementation details
- Better metadata and documentation
- Updated author information
- Fixed issue with pipeline implementation privacy

### Key features
- Hybrid retrieval: BM25 over rich function docs; optional semantic fusion
- Strict top‑k seed selection, BFS expansion over callers/callees
- Identifier TF‑IDF, PageRank on call graph, structure/complexity features
- CAST chunking: byte‑safe, newline‑aligned split/merge with token caps
- JSON payload with selected functions/classes and optional code blob

---

## Installation

### From PyPI (Recommended)
```bash
pip install HasteContext==0.2.1
```

Visit the package on PyPI: [https://pypi.org/project/HasteContext/0.2.1/](https://pypi.org/project/HasteContext/0.2.1/)

### Using Poetry
```bash
poetry add HasteContext
```

### Development Installation
```bash
git clone https://github.com/Hacxmr/AST-Relevance-Compression.git
cd AST-Relevance-Compression
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -e .
```

Python 3.11+ is required. Core runtime dependencies include:
- `tree-sitter`
- `tree-sitter-language-pack`
- `tiktoken`
- `numpy`
- `rank-bm25`
- `openai`

Optional: set your OpenAI API key when using semantic reranking or embeddings-backed flows.
```bash
# Windows Command Prompt
set OPENAI_API_KEY=your_key_here

# Windows PowerShell
$env:OPENAI_API_KEY = "your_key_here"
```

---

## Quickstart (programmatic)

Use the single-import public API facade for end-to-end flows:

```python
from haste import select_from_file, build_payload_from_repo

# Single file, mirrors CLI output structure (nodes/classes/selected/code)
out = select_from_file(
    "path/to/file.py",
    query="find dataloader and training loop",
    top_k=6,
    bfs_depth=1,
)
print(out["nodes"][:2])
print(out["code"][:500])

# Repository-level payload (index the tree and select relevant code)
payload = build_payload_from_repo(
    "path/to/repo",
    include_code=True,
    top_k=50,
    depth=1,
    query="http handler metrics",
)
```

This reduces import boilerplate and keeps a stable, public surface.

---

## CLI (single Python file)

The minimal CLI operates on a single `.py` file and prints JSON.

```bash
hastecontext path\to\file.py --query "find dataloader and training loop" \
  --top-k 6 --prefilter 300 --bfs-depth 1 --max-add 12 \
  --hard-cap 1200 --soft-cap 1800 [--semantic] [--sem-model text-embedding-3-small]
```

Flags:
- `--query` (required): free‑form text
- `--top-k`: seed size (default 6)
- `--prefilter`: lexical candidate pool before rerank (default 300)
- `--bfs-depth`: expansion hops over same‑module call edges (default 1)
- `--max-add`: cap on nodes added by BFS (default 12)
- `--semantic`: enable OpenAI embeddings rerank (requires `OPENAI_API_KEY`)
- `--sem-model`: embeddings model (default `text-embedding-3-small`)
- `--hard-cap`, `--soft-cap`: CAST token caps used during chunk split/merge

Example output shape:
```json
{
  "summary": {"total_functions": 12, "total_classes": 3},
  "nodes": [ {"type": "function", "name": "train", "qname": "module::train", "path": "...", "lineno": 10, "end_lineno": 120, "signature": "train(cfg)", "docstring": "...", "score": 0.71} ],
  "classes": [ {"type": "class", "name": "DataLoader", "qname": "module::DataLoader", "path": "..."} ],
  "selected": {"roots": ["module::train"], "functions": ["module::train", "module::step"], "classes": ["module::DataLoader"]},
  "code": "...stitched code under token caps..."
}
```

Also runnable from source without installing the script:
```bash
python -m haste.cli path\to\file.py --query "..."
```

You can also use the installed console script:
```bash
hastecontext path\to\file.py --query "..."
```

---

## Advanced usage (lower-level building blocks)

If you need full control, the lower-level modules remain available (indexing, metrics, selection, assembly). See `haste.index`, `haste.metrics`, and `haste.selection` for granular APIs.

---

## How it works
1) Index with Tree‑sitter: collect functions/classes, call edges, decorators, docstrings, variables, and module API hints
2) Score: compute PageRank on the call graph; TF‑IDF over identifiers; cyclomatic complexity and structure richness
3) Retrieve: BM25 over rich function docs; optionally fuse semantic rankings via embeddings + RRF
4) Select: enforce strict top‑k seeds; expand via BFS over callers/callees; re‑rank by fused score
5) Compress: CAST split/merge spans with hard/soft token caps; stitch to a contiguous code blob

---

## Requirements & Compatibility
- Python 3.11, 3.12, 3.13
- Tree‑sitter runtime and `tree-sitter-language-pack` for Python
- OpenAI API key only needed for `--semantic` or when using `OpenAIEmbedder`
- All major operating systems supported (Windows, macOS, Linux)
- This package does not include `pipeline.py`, `reports/`, and test scripts, which are used only for internal metrics.

---

## Contributing
PRs welcome. Use Poetry for the dev environment (`poetry install`). Run linters/formatters as you normally would; keep public API changes minimal and documented.


## License
MIT. See LICENSE file in the repository root. 
