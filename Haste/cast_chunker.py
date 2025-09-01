# SPDX-License-Identifier: MIT
# CAST: byte-safe, newline-aligned split-then-merge with token budgeting.

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import tiktoken  # type: ignore


@dataclass(frozen=True)
class ByteSpan:
    start: int  # inclusive byte offset
    end: int    # exclusive byte offset


def _token_counter():
    enc = tiktoken.get_encoding("cl100k_base")

    def count(blob: bytes) -> int:
        # decode with replacement for robust counting; only used for token count
        return len(enc.encode(blob.decode("utf-8", errors="replace")))

    return count


def _find_boundaries(src_bytes: bytes, start: int, end: int) -> List[int]:
    """Return safe newline boundaries in [start, end]. Always include end."""
    b = src_bytes
    bounds: List[int] = []

    i = start
    # prefer double-newlines
    while True:
        j = b.find(b"\n\n", i, end)
        if j == -1:
            break
        bounds.append(j + 2)
        i = j + 2

    if not bounds:
        i = start
        while True:
            j = b.find(b"\n", i, end)
            if j == -1:
                break
            bounds.append(j + 1)
            i = j + 1

    if not bounds or bounds[-1] != end:
        bounds.append(end)
    return bounds


def _split_one_span(src_bytes: bytes, sp: ByteSpan, max_tokens: int, count_tokens) -> List[ByteSpan]:
    piece_tokens = count_tokens(src_bytes[sp.start:sp.end])
    if piece_tokens <= max_tokens:
        return [sp]

    bounds = _find_boundaries(src_bytes, sp.start, sp.end)
    pieces: List[ByteSpan] = []
    cur_start = sp.start
    cur_tokens = 0
    last_ok = cur_start

    for fence in bounds:
        if fence <= cur_start:
            continue
        t = count_tokens(src_bytes[cur_start:fence])
        if cur_tokens + t > max_tokens and fence != bounds[0]:
            pieces.append(ByteSpan(cur_start, last_ok))
            cur_start = last_ok
            cur_tokens = count_tokens(src_bytes[cur_start:fence])
        else:
            cur_tokens += t
        last_ok = fence

    pieces.append(ByteSpan(cur_start, bounds[-1]))

    # sanity: contiguous coverage
    assert pieces[0].start == sp.start and pieces[-1].end == sp.end
    for i in range(1, len(pieces)):
        assert pieces[i - 1].end == pieces[i].start
    return pieces


def cast_split_merge(src_bytes: bytes,
                     spans: List[ByteSpan],
                     hard_cap_tokens: int,
                     soft_cap_tokens: int) -> List[ByteSpan]:
    """
    1) Split any span exceeding hard_cap_tokens at safe newline fences.
    2) Greedily merge adjacent spans while staying under soft_cap_tokens.
    All operations are done on byte offsets to avoid mid-codepoint cuts.
    """
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    count_tokens = _token_counter()

    # Step 1: split large spans
    small_spans: List[ByteSpan] = []
    for sp in spans:
        small_spans.extend(_split_one_span(src_bytes, sp, hard_cap_tokens, count_tokens))

    if not small_spans:
        return []

    # Step 2: greedy merge respecting soft cap
    merged: List[ByteSpan] = []
    cur = small_spans[0]
    cur_tokens = count_tokens(src_bytes[cur.start:cur.end])

    for nxt in small_spans[1:]:
        if cur.end != nxt.start:
            # Non-adjacent; flush current
            merged.append(cur)
            cur = nxt
            cur_tokens = count_tokens(src_bytes[cur.start:cur.end])
            continue

        nxt_tokens = count_tokens(src_bytes[nxt.start:nxt.end])
        if cur_tokens + nxt_tokens <= soft_cap_tokens:
            cur = ByteSpan(cur.start, nxt.end)
            cur_tokens += nxt_tokens
        else:
            merged.append(cur)
            cur = nxt
            cur_tokens = nxt_tokens

    merged.append(cur)
    return merged


