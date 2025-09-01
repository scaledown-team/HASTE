from typing import List
import os
import numpy as np
from openai import OpenAI 

class OpenAIEmbedder:
    """
    Thin wrapper around OpenAI embeddings API returning L2-normalized numpy vectors.

    Requires OPENAI_API_KEY in environment. Model defaults to text-embedding-3-small.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI embeddings.")
        self.client = OpenAI()
        self.model_name = model_name

    def encode(self, texts: List[str]):
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        mats = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(mats, axis=1, keepdims=True) + 1e-12
        return (mats / norms)