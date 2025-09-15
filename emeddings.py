#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a FAISS index from chunks_clean_final.jsonl
Embeddings: Gemini (default) or Sentence-Transformers (local)

Usage:
  python embeddings_faiss_gemini.py --chunks Data/processed/chunks_clean_final.jsonl ^
                                    --persist Data/index/faiss ^
                                    --collection vit_faq_vellore ^
                                    --emb gemini

  # Or local embeddings (no API):
  python embeddings_faiss_gemini.py --chunks Data/processed/chunks_clean_final.jsonl ^
                                    --persist Data/index/faiss ^
                                    --collection vit_faq_vellore ^
                                    --emb st
"""
import os, sys, json, argparse
from pathlib import Path
from tqdm import tqdm

# Only use the community import (no deprecation warning)
from langchain_community.vectorstores import FAISS

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------------- Embedding backends ---------------- #

class GeminiEmbeddings:
    """
    Minimal adapter for Gemini embeddings.
    Uses the CURRENT model: models/text-embedding-004
    """
    def __init__(self, model: str = "models/text-embedding-004", api_key_env: str = "GEMINI_API_KEY"):
        import google.generativeai as genai
        api_key = os.getenv(api_key_env)
        if not api_key:
            print("[ERROR] GEMINI_API_KEY not set. Set it via environment or .env file.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = model

    def embed_documents(self, texts):
        out = []
        for t in texts:
            # embed_content returns {'embedding': [...]} for this model
            r = self.genai.embed_content(model=self.model, content=t)
            vec = r.get("embedding")
            if vec is None:
                raise RuntimeError("Gemini embed_content returned no 'embedding'. Check model & quota.")
            out.append(vec)
        return out

    def embed_query(self, q):
        r = self.genai.embed_content(model=self.model, content=q)
        vec = r.get("embedding")
        if vec is None:
            raise RuntimeError("Gemini embed_content returned no 'embedding' for query.")
        return vec


class STEmbeddings:
    """Sentence-Transformers local embeddings (no API)."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            print("[ERROR] sentence-transformers not installed. pip install sentence-transformers")
            raise
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, q):
        return self.model.encode([q], normalize_embeddings=True)[0].tolist()


# ---------------- Utilities ---------------- #

def load_chunks(jsonl_path: Path):
    texts, metas, ids = [], [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            txt = (obj.get("text") or "").strip()
            if not txt:
                continue
            texts.append(txt)
            metas.append(obj.get("metadata", {}) or {})
            ids.append(obj.get("id") or f"id_{len(ids)}")
    return texts, metas, ids


# ---------------- Build FAISS ---------------- #

def build_faiss(chunks, persist_dir: Path, collection: str, emb):
    texts, metas, ids = chunks
    if not texts:
        print("[ERROR] No texts to index. Is your chunks file empty?")
        sys.exit(1)

    print(f"[INFO] Building FAISS index with {len(texts)} docs (this may take a minute)...")
    # langchain_community FAISS expects .embed_documents/.embed_query on embedding object
    vectors = emb.embed_documents(texts)

    # Create store from precomputed embeddings to avoid re-embedding inside FAISS
    # (this path avoids issues if your embedding object isn't a LangChain Embeddings subclass)
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_core.documents import Document
    import faiss
    import numpy as np
    import uuid

    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors (Gemini returns L2; we keep IP; normalize if needed)
    # normalize embeddings for cosine similarity
    vec_np = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(vec_np, axis=1, keepdims=True) + 1e-12
    vec_np = vec_np / norms
    index.add(vec_np)

    # Build documents & ID map
    docs = {}
    id_map = {}
    for i, (t, m) in enumerate(zip(texts, metas)):
        did = ids[i] if i < len(ids) else str(uuid.uuid4())
        docs[did] = Document(page_content=t, metadata=m)
        id_map[i] = did

    store = FAISS(
        embedding_function=emb,  # kept for API compatibility; not used since we precomputed
        index=index,
        docstore=InMemoryDocstore(docs),
        index_to_docstore_id=id_map,
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(persist_dir), index_name=collection)
    print(f"[DONE] FAISS saved at: {persist_dir}/{collection}.faiss (+ .pkl)")


# ---------------- Main ---------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Path to Data/processed/chunks_clean_final.jsonl")
    ap.add_argument("--persist", required=True, help="Folder to save FAISS (e.g., Data/index/faiss)")
    ap.add_argument("--collection", default="vit_faq_vellore", help="Index name")
    ap.add_argument("--emb", choices=["gemini","st"], default="gemini", help="Embedding backend")
    args = ap.parse_args()

    chunks_path = Path(args.chunks).resolve()
    if not chunks_path.exists():
        print(f"[ERROR] chunks not found: {chunks_path}")
        sys.exit(1)

    if args.emb == "gemini":
        emb = GeminiEmbeddings()  # uses models/text-embedding-004
    else:
        emb = STEmbeddings()

    texts, metas, ids = load_chunks(chunks_path)
    build_faiss((texts, metas, ids), Path(args.persist).resolve(), args.collection, emb)
