#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Query tokenization and expansion utilities.

Adds generic synonyms, light morphology, and camel/snake splitting.
"""

import re
from typing import List


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

GENERIC_SYNONYMS = {
	"lr": ["learning", "rate", "learningrate"],
	"step": ["steps", "iteration", "iter", "epoch", "update"],
	"train": ["trainer", "training", "fit", "optimize"],
	"batch": ["batches", "minibatch", "minibatches"],
	"save": ["checkpoint", "dump", "persist", "serialize"],
	"eval": ["evaluate", "evaluation", "test", "metric", "validation"],
	"loader": ["dataloader", "iterator", "dataset"],
	"config": ["cfg", "args", "hyperparam", "hyperparameters"],
	"schedule": ["scheduler", "warmup", "cosine", "anneal", "decay"],
}


def split_terms(s: str) -> List[str]:
	parts = _WORD.findall((s or "").lower())
	extra: List[str] = []
	for p in parts:
		# split camel and snake
		p2 = re.sub(r"([a-z])([A-Z])", r"\1 \2", p)
		p2 = re.sub(r"[_]+", " ", p2)
		extra += re.findall(r"[a-z]+", p2)
	return list({*(parts + extra)})


def expand_query(q: str) -> List[str]:
	toks = split_terms(q)
	expanded = set(toks)
	for t in toks:
		if t in GENERIC_SYNONYMS:
			expanded.update(GENERIC_SYNONYMS[t])
		# very light morphology
		if t.endswith("s"): expanded.add(t[:-1])
		if t.endswith("ing"): expanded.add(t[:-3])
		if t.endswith("ed"): expanded.add(t[:-2])
	return list(expanded)


