#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a vector index from chunks_clean_final.jsonl
Embeddings: Gemini (default) or Sentence-Transformers
DB: Chroma (default) or FAISS

Usage:
  # Chroma + Gemini embeddings (recommended)
  python build_index_gemini.py --chunks data/processed/chunks_clean_final.jsonl ^
                               --db chroma --persist data/index/chroma ^
                               --collection vit_faq_vellore --emb gemini

  # Chroma + local ST embeddings (fallback, no API)
  python build_index_gemini.py --chunks data/processed/chunks_clean_final.jsonl ^
                               --db chroma --persist data/index/chroma ^
                               --collection vit_faq_vellore --emb st

  # FAISS + Gemini (if faiss-cpu installed)
  python build_index_gemini.py --chunks data/processed/chunks_clean_final.jsonl ^
                               --db faiss --persist data/index/faiss ^
                               --collection vit_faq_vellore --emb gemini
"""
import os, sys, json, argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS


# --- Embedding backends ---
class GeminiEmbeddings:
    """Minimal adapter for Gemini text-embedding-004."""
    def __init__(self, api_key_env="GEMINI_API_KEY", model="text-embedding-001"):
        import google.generativeai as genai
        api_key = os.getenv(api_key_env)
        if not api_key:
            print("[ERROR] GEMINI_API_KEY not set.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = model

    def embed_documents(self, texts):
        # batch to be gentle on rate limits
        out = []
        for t in texts:
            r = self.genai.embed_content(model=self.model, content=t)
            out.append(r["embedding"])
        return out

    def embed_query(self, q):
        r = self.genai.embed_content(model=self.model, content=q)
        return r["embedding"]

class STEmbeddings:
    """Sentence-Transformers local embeddings (no API)."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, q):
        return self.model.encode([q], normalize_embeddings=True)[0].tolist()

def load_chunks(jsonl_path: Path):
    texts, metas, ids = [], [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            txt = (obj.get("text") or "").strip()
            if not txt: continue
            texts.append(txt)
            metas.append(obj.get("metadata", {}) or {})
            ids.append(obj.get("id") or f"id_{len(ids)}")
    return texts, metas, ids

# ---------------- CHROMA path (direct client, no LangChain) ---------------- #
def build_chroma(chunks, persist_dir: Path, collection: str, emb):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))

    # create or get collection (cosine distance)
    try:
        col = client.get_collection(collection)
    except Exception:
        col = client.create_collection(collection, metadata={"hnsw:space":"cosine"})

    texts, metas, ids = chunks
    print(f"[INFO] adding {len(texts)} docs to Chroma collection '{collection}'")

    B = 256
    for i in tqdm(range(0, len(texts), B), desc="Indexing → Chroma"):
        batch_texts = texts[i:i+B]
        batch_ids   = ids[i:i+B]
        batch_meta  = metas[i:i+B]
        batch_embs  = emb.embed_documents(batch_texts)
        col.add(documents=batch_texts, embeddings=batch_embs, metadatas=batch_meta, ids=batch_ids)
    print(f"[DONE] Chroma persisted at: {persist_dir}")

# ---------------- FAISS path (via LangChain) ---------------- #
def build_faiss(chunks, persist_dir: Path, collection: str, emb):
    try:
        from langchain.vectorstores import FAISS
    except Exception as e:
        print("[ERROR] langchain FAISS not available. Install langchain & faiss-cpu.", e)
        sys.exit(1)

    texts, metas, ids = chunks
    print(f"[INFO] building FAISS index: {len(texts)} docs")
    vs = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metas, ids=ids)
    persist_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(persist_dir), index_name=collection)
    print(f"[DONE] FAISS saved at: {persist_dir}/{collection}.faiss + .pkl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="data/processed/chunks_clean_final.jsonl")
    ap.add_argument("--db", choices=["chroma","faiss"], default="chroma")
    ap.add_argument("--persist", required=True, help="folder for the index")
    ap.add_argument("--collection", default="vit_faq_vellore")
    ap.add_argument("--emb", choices=["gemini","st"], default="gemini")
    args = ap.parse_args()

    chunks_path = Path(args.chunks).resolve()
    if not chunks_path.exists():
        print(f"[ERROR] chunks not found: {chunks_path}"); sys.exit(1)

    if args.emb == "gemini":
        emb = GeminiEmbeddings()
    else:
        emb = STEmbeddings()

    texts, metas, ids = load_chunks(chunks_path)
    chunks = (texts, metas, ids)

    if args.db == "chroma":
        build_chroma(chunks, Path(args.persist).resolve(), args.collection, emb)
    else:
        build_faiss(chunks, Path(args.persist).resolve(), args.collection, emb)



