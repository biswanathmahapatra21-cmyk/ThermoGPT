import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from PyPDF2 import PdfReader
from textwrap import wrap


def extract_text_from_pdfs(folder):
    all_texts = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".pdf"):
            pdf = PdfReader(os.path.join(folder, fname))
            text = " ".join(page.extract_text() or "" for page in pdf.pages)
            chunks = wrap(text, 500)  # break into manageable chunks
            all_texts.extend(chunks)
    return all_texts


def build_knowledge_base(data_folder="data/corpus", save_folder="embeddings"):
    model = SentenceTransformer("all-mpnet-base-v2")
    texts = extract_text_from_pdfs(data_folder)
    embs = model.encode(texts, convert_to_numpy=True)

    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "embs.npy"), embs)
    with open(os.path.join(save_folder, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    nn = NearestNeighbors(n_neighbors=4).fit(embs)
    print(f"✅ Knowledge base built with {len(texts)} chunks.")
    return nn, texts, embs


if __name__ == "__main__":
    build_knowledge_base()
