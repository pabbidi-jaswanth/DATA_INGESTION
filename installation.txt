# app/utils/embeddings.py
# Single, offline TF-IDF embedding backend.
# We persist a vectorizer per collection directory and always load it at query time.

from __future__ import annotations

import pathlib
import pickle
from typing import List
import numpy as np


class TFIDFEncoder:
    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer  # fitted TfidfVectorizer

    @property
    def is_transformer(self) -> bool:
        return False

    def fit(self, texts: List[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
        )
        self.vectorizer.fit(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF encoder not fitted.")
        X = self.vectorizer.transform(texts)
        X = X.astype(np.float32).toarray()
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
        return X

    def save(self, dest_dir: pathlib.Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

    @staticmethod
    def load(dest_dir: pathlib.Path) -> "TFIDFEncoder":
        p = dest_dir / "vectorizer.pkl"
        if not p.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {p}. Build indexes first."
            )
        with open(p, "rb") as f:
            vec = pickle.load(f)
        return TFIDFEncoder(vectorizer=vec)


def get_encoder_for_build():
    """
    Build-time: always TF-IDF (offline).
    """
    return TFIDFEncoder()


def get_encoder_for_query(collection_dir: pathlib.Path):
    """
    Query-time: always load TF-IDF vectorizer from the collection dir.
    """
    return TFIDFEncoder.load(collection_dir)
