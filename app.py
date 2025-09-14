#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json
from pathlib import Path
import streamlit as st
import google.generativeai as genai

# ---------------- keyword → metadata filters ---------------- #
DOMAIN_MAP = {
    "ug":"UG","undergrad":"UG",
    "pg":"PG","postgrad":"PG","post graduate":"PG","post-graduate":"PG",
    "hostel":"Hostel","accommodation":"Hostel",
    "research":"Research","phd":"Research","vitree":"Research",
}
PROGRAM_MAP = {
    "mca":"MCA","m.tech":"M.Tech","mtech":"M.Tech","msc":"M.Sc","m.sc":"M.Sc",
    "btech":"B.Tech","b.tech":"B.Tech","phd":"Ph.D"
}
CATEGORY_MAP = {
    "refund policy":"Refund Policy","refund":"Refund Policy",
    "fee structure":"Fee Structure","fees":"Fee Structure","tuition":"Fee Structure",
    "courses offered":"Courses Offered","programs offered":"Courses Offered","programmes offered":"Courses Offered",
    "eligibility criteria":"Eligibility Criteria","eligibility":"Eligibility Criteria",
    "documents required":"Documents Required","documents":"Documents Required","document submission":"Documents Required",
    "academic rules":"Academic Rules","attendance":"Academic Rules","grading":"Academic Rules","ffcs":"Academic Rules",
    "hostel norms":"Hostel Norms","norms":"Hostel Norms",
    "admission process":"Admission Process","process":"Admission Process","procedure":"Admission Process",
    "exam pattern":"Exam Pattern","faq":"FAQ","faqs":"FAQ",
    "contacts":"Contacts","contact":"Contacts","helpline":"Contacts"
}
AUDIENCE_MAP = {"nri":"NRI","foreign":"Foreign","international":"Foreign"}

def build_filters_from_query(q: str):
    ql = q.lower()
    where = {}
    for k,v in DOMAIN_MAP.items():
        if k in ql: where["domain"] = v; break
    for k,v in PROGRAM_MAP.items():
        if k in ql: where["program"] = v; break
    for k,v in CATEGORY_MAP.items():
        if k in ql: where["category"] = v; break
    for k,v in AUDIENCE_MAP.items():
        if k in ql: where["audience"] = v; break
    where.setdefault("campus","Vellore")   # MVP default
    return where

# ---------------- Embedding backends ---------------- #
class GeminiEmbeddings:
    def __init__(self, model="text-embedding-004"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
    def embed_documents(self, texts):
        out = []
        for t in texts:
            r = genai.embed_content(model=self.model, content=t)
            out.append(r["embedding"])
        return out
    def embed_query(self, q):
        return genai.embed_content(model=self.model, content=q)["embedding"]

class STEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, q):
        return self.model.encode([q], normalize_embeddings=True)[0].tolist()

# ---------------- Vector stores ---------------- #
def open_chroma(persist_dir: str, collection: str):
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(collection)

def query_chroma(col, q: str, where: dict, emb) -> list:
    q_emb = emb.embed_query(q)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=st.session_state.get("top_k", 5),
        where=where,
        include=["documents","metadatas","distances"],
    )
    docs = []
    if res and res.get("documents"):
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            docs.append({"text": doc, "meta": meta, "score": 1.0 - dist})  # cosine -> similarity
    return docs

def open_faiss(persist_dir: str, collection: str, emb):
    from langchain.vectorstores import FAISS
    folder = Path(persist_dir)
    return FAISS.load_local(folder, embeddings=emb, index_name=collection, allow_dangerous_deserialization=True)

def query_faiss(vs, q: str, where: dict) -> list:
    # LangChain FAISS doesn’t support server-side metadata filters → filter client-side
    docs = vs.similarity_search_with_score(q, k=st.session_state.get("top_k", 5))
    out = []
    for d, dist in docs:
        md = d.metadata or {}
        ok = all(md.get(k) == v for k,v in where.items())
        if ok:
            out.append({"text": d.page_content, "meta": md, "score": 1.0 - dist})
    return out

# ---------------- Generation (Gemini 1.5 Flash) ---------------- #
def generate_answer(context_docs: list, question: str) -> str:
    if not os.getenv("GEMINI_API_KEY"):
        return ""  # generation disabled
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Build grounded context (truncate a bit)
    ctx = []
    for d in context_docs[:4]:
        meta = d["meta"]
        src = meta.get("source_file") or meta.get("source_title") or ""
        snippet = d["text"]
        if len(snippet) > 900: snippet = snippet[:900] + "..."
        ctx.append(f"[{src}] {snippet}")
    context = "\n\n".join(ctx)

    prompt = f"""You are a helpful assistant answering VIT Vellore admissions/academics/hostel queries.
Use ONLY the context below. Be concise (2–4 sentences).
If fees/dates/hostel fees are involved, include AY if present and say: "Info may change—check source".

Context:
{context}

Question: {question}
Answer (with 1–2 inline source tags like [filename.pdf]):"""

    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip()

# ---------------- UI ---------------- #
st.set_page_config(page_title="VIT Smart FAQ (RAG, Gemini Flash)", page_icon="🎓", layout="wide")
st.title("🎓 VIT Smart FAQ — RAG (Gemini 1.5 Flash)")
st.caption("Single chatbox → keyword-aware filters (UG/PG/MCA/NRI/etc.) → retrieve → (optional) generate")

with st.sidebar:
    st.header("Settings")
    db_type = st.selectbox("Vector DB", ["Chroma","FAISS"], index=0)
    emb_type = st.selectbox("Embeddings", ["Gemini (text-embedding-004)","Sentence-Transformers"], index=0)
    st.session_state["top_k"] = st.slider("Top-K", 3, 10, 5)
    persist_dir = st.text_input("Index path", value="data/index/chroma" if db_type=="Chroma" else "data/index/faiss")
    collection = st.text_input("Collection / Index name", value="vit_faq_vellore")
    gen_enabled = st.checkbox("Generate with Gemini Flash (else show snippets)", value=bool(os.getenv("GEMINI_API_KEY")))

# init embeddings
if emb_type.startswith("Gemini"):
    emb = GeminiEmbeddings()
else:
    emb = STEmbeddings()

# load vector store
try:
    if db_type == "Chroma":
        col = open_chroma(persist_dir, collection)
        vs = ("chroma", col)
    else:
        vs = ("faiss", open_faiss(persist_dir, collection, emb))
except Exception as e:
    st.error(f"Failed to open index: {e}")
    st.stop()

# chat
q = st.text_input("Ask your question (e.g., 'pg mca refund policy', 'ug nri fees', 'hostel refund policy')", "")
if q:
    where = build_filters_from_query(q)

    if vs[0] == "chroma":
        results = query_chroma(vs[1], q, where, emb)
    else:
        results = query_faiss(vs[1], q, where)

    if not results:
        st.warning("No results found for the current filters. Try rephrasing or removing some keywords.")
    else:
        # show generated answer OR grounded snippets
        if gen_enabled:
            answer = generate_answer(results, q)
            if answer:
                st.subheader("Answer")
                st.write(answer)
            else:
                st.info("Generation unavailable—showing grounded snippets instead.")

        st.subheader("Top results")
        for i, d in enumerate(results, 1):
            md = d["meta"]; src = md.get("source_file"); tag = f"{md.get('domain')}/{md.get('category')}/{md.get('program')}"
            with st.expander(f"{i}. {src}  [{tag}]  score={d['score']:.3f}"):
                st.write(d["text"])
                st.caption(md)
