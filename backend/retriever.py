"""
Final, stable Retriever for ThermalGPT
Loads embeddings from ingest_corpus.py
Always sets self.nn (no AttributeError)
Windows-safe absolute paths
"""

import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self):
        print("🔍 Initializing Retriever...")

        # ---------- ROOT PATH (Windows-safe) ----------
        self.ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        print("📁 Project Root:", self.ROOT)

        # ---------- Embedding paths ----------
        self.texts_path = os.path.join(self.ROOT, "embeddings", "texts.pkl")
        self.embs_path = os.path.join(self.ROOT, "embeddings", "embs.npy")

        print("📄 Looking for texts at:", self.texts_path)
        print("📄 Looking for embeddings at:", self.embs_path)

        # ---------- initialize attributes ----------
        self.texts = []
        self.embs = None
        self.nn = None  # ALWAYS exists

        # ---------- Model ----------
        self.model = SentenceTransformer("all-mpnet-base-v2")

        # ---------- Try loading the real corpus ----------
        self._load_existing_embeddings()


    # ------------------------------------------------------------
    def _load_existing_embeddings(self):
        """Load PDF-based corpus if available"""
        if os.path.exists(self.texts_path) and os.path.exists(self.embs_path):
            print("📘 Loading existing corpus...")
            try:
                with open(self.texts_path, "rb") as f:
                    self.texts = pickle.load(f)

                self.embs = np.load(self.embs_path)
                self.nn = NearestNeighbors(n_neighbors=4).fit(self.embs)

                print(f"✅ Corpus loaded with {len(self.texts)} chunks!")
            except Exception as e:
                print("❌ Failed to load embeddings:", e)
                self.nn = None
        else:
            print("⚠️ No corpus found — nn will remain None until built")


    # ------------------------------------------------------------
    def build(self, texts):
        """Build a new index from provided text list (fallback only)."""
        print("⚙️ Building fallback sample index...")

        self.texts = texts
        self.embs = self.model.encode(texts, convert_to_numpy=True)

        os.makedirs(os.path.join(self.ROOT, "embeddings"), exist_ok=True)
        np.save(self.embs_path, self.embs)

        with open(self.texts_path, "wb") as f:
            pickle.dump(texts, f)

        self.nn = NearestNeighbors(n_neighbors=min(4, len(self.embs))).fit(self.embs)
        print(f"✅ Fallback index built with {len(texts)} chunks.")


    # ------------------------------------------------------------
    def retrieve(self, query, k=3):
        if self.nn is None:
            raise RuntimeError(
                "❌ No index found.\n"
                "Run `python backend/ingest_corpus.py` to create real embeddings."
            )

        q_emb = self.model.encode([query], convert_to_numpy=True)
        dists, idxs = self.nn.kneighbors(q_emb, n_neighbors=min(k, len(self.texts)))

        return [self.texts[i] for i in idxs[0]]


if __name__ == "__main__":
    r = Retriever()
    if r.nn:
        print(r.retrieve("What is convection heat transfer?", k=3))
    else:
        print("❌ No index found.")

