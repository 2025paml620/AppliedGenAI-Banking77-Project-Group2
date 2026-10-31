"""RAG retriever: finds similar labelled BANKING77 *train* examples for a query.

The corpus is the official train split only, so evaluation (test split) queries and
labels never leak into the prompt.
"""
from typing import Callable, Sequence

import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, texts: Sequence[str], intents: Sequence[str], encode: Callable[[list[str]], np.ndarray]):
        """encode(list of str) -> 2D array; vectors are normalised so a dot product is cosine similarity."""
        if len(texts) != len(intents) or not len(texts):
            raise ValueError("texts and intents must be nonempty and the same length")
        self.texts, self.intents, self.encode = list(texts), list(intents), encode
        self.matrix = self._normalise(encode(self.texts))

    @staticmethod
    def _normalise(vectors) -> np.ndarray:
        """Scale each embedding to unit length while safely handling zero vectors."""
        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Return up to k nearest training examples, ordered by cosine similarity."""
        scores = self.matrix @ self._normalise(self.encode([query]))[0]
        top = np.argsort(-scores)[:k]
        return [{"text": self.texts[i], "intent": self.intents[i], "score": float(scores[i])} for i in top]


def build_retriever(device: str | None = None) -> Retriever:
    """Embed the BANKING77 train split with a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    from src.data.dataset import load_banking77

    train = load_banking77()["train"]
    names = train.features["label"].names
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    encode = lambda texts: embedder.encode(texts, batch_size=128, show_progress_bar=False)
    return Retriever(train["text"], [names[i] for i in train["label"]], encode)
