# SPDX-License-Identifier: MIT
# Robust exporter that ALWAYS returns non-empty `code` (fixes the empty-code bug).

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ByteSpan:
    start: int
    end: int


def stitch_code(src_bytes: bytes, spans: List[ByteSpan]) -> tuple[str, List[tuple[int, int, int, int]]]:
    """
    Returns:
      code (str): concatenated slices separated by one blank line
      mapping: list of tuples (span_index, src_start_line, src_end_line, out_start_line)
    We compute line numbers from bytes to keep it robust across encodings.
    """
    if not spans:
        # Export entire file as last-resort (never empty)
        text = src_bytes.decode("utf-8", errors="replace")
        return text, [(0, 1, text.count("\n") + 1, 1)]

    # Sort & coalesce
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    parts: List[str] = []
    mapping: List[tuple[int, int, int, int]] = []
    out_line = 1
    src_prefix_lines = _prefix_line_numbers(src_bytes)
    for i, sp in enumerate(spans):
        chunk = src_bytes[sp.start:sp.end].decode("utf-8", errors="replace")
        start_line = _byte_to_line(src_prefix_lines, sp.start)
        end_line = _byte_to_line(src_prefix_lines, sp.end)
        parts.append(chunk)
        mapping.append((i, start_line, end_line, out_line))
        out_line += chunk.count("\n") + 1  # +1 for the added blank line below
        parts.append("\n")
    code = "".join(parts).rstrip("\n")
    return code, mapping


def _prefix_line_numbers(src: bytes) -> List[int]:
    # prefix sum of newline byte positions (plus 0 at start)
    line_starts = [0]
    for i, b in enumerate(src):
        if b == 10:  # '\n'
            line_starts.append(i + 1)
    # add end sentinel to simplify edge cases
    line_starts.append(len(src))
    return line_starts


def _byte_to_line(prefix: List[int], byte_pos: int) -> int:
    # binary search for the rightmost line_start <= byte_pos
    lo, hi = 0, len(prefix) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if prefix[mid] <= byte_pos:
            lo = mid
        else:
            hi = mid - 1
    return max(1, lo + 1)  # 1-based lines














