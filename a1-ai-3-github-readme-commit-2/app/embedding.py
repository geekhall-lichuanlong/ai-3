import hashlib
import math
import re

import numpy as np

from app.config import EMBEDDING_DIMS


LATIN_RE = re.compile(r"[a-zA-Z0-9_]+", re.UNICODE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in LATIN_RE.findall(text)]
    cjk_chars = CJK_RE.findall(text)
    tokens.extend(cjk_chars)
    tokens.extend("".join(pair) for pair in zip(cjk_chars, cjk_chars[1:], strict=False))
    return tokens


class HashingEmbedding:
    """Deterministic local embedding for offline demos and stable tests."""

    def __init__(self, dims: int = EMBEDDING_DIMS) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dims, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return vector.tolist()
        return (vector / norm).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
