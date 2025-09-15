#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG answering for VIT Vellore chatbot
- loads FAISS index you already built
- retrieves top-k chunks (with metadata-aware post-filter)
- uses Gemini 1.5 Flash to extract exact facts into a TABLE (when fees/hostel)
- renders a clean Markdown answer with a short summary + sources

Usage (Windows PowerShell):
  .\.venv\Scripts\python.exe rag_answer.py `
    --index_dir Data/index/faiss `
    --collection vit_faq_vellore `
    --emb gemini `
    --q "ug nri ladies hostel fees 2025"

Prereqs:
  pip install google-generativeai langchain-community faiss-cpu python-dotenv
Env:
  setx GEMINI_API_KEY "YOUR_KEY_HERE"
"""

import os, sys, json, argparse, textwrap, re
from pathlib import Path
from typing import List, Dict, Any, Optional

# FAISS (langchain community)
from langchain_community.vectorstores import FAISS

# try loading .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------- Embeddings (same as your index) -------------------- #
class GeminiEmbeddings:
    """Adapter for Gemini embeddings (same family used during indexing)."""
    def __init__(self, model: str = "models/text-embedding-004"):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[ERROR] GEMINI_API_KEY not set."); sys.exit(1)
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = model
    def embed_documents(self, texts: List[str]):
        out = []
        for t in texts:
            r = self.genai.embed_content(model=self.model, content=t)
            out.append(r["embedding"])
        return out
    def embed_query(self, q: str):
        r = self.genai.embed_content(model=self.model, content=q)
        return r["embedding"]

class STEmbeddings:
    """Local fallback (not needed if you use gemini)."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, q: str):
        return self.model.encode([q], normalize_embeddings=True)[0].tolist()

# -------------------- Simple query→metadata hints (router) -------------------- #
DOMAIN_MAP = {
    "ug":"UG", "undergrad":"UG",
    "pg":"PG", "postgrad":"PG", "post graduate":"PG",
    "hostel":"Hostel", "accommodation":"Hostel",
    "research":"Research", "phd":"Research", "vitree":"Research",
}
PROGRAM_MAP = {
    "mca":"MCA", "m.tech":"M.Tech", "mtech":"M.Tech",
    "msc":"M.Sc", "m.sc":"M.Sc", "btech":"B.Tech","b.tech":"B.Tech","phd":"Ph.D"
}
CATEGORY_MAP = {
    "refund policy":"Refund Policy", "refund":"Refund Policy",
    "fee structure":"Fee Structure", "fees":"Fee Structure","tuition":"Fee Structure","hostel fee":"Fee Structure",
    "courses offered":"Courses Offered","programs offered":"Courses Offered","programmes offered":"Courses Offered","courses":"Courses Offered",
    "eligibility criteria":"Eligibility Criteria","eligibility":"Eligibility Criteria",
    "documents required":"Documents Required","documents":"Documents Required","document submission":"Documents Required",
    "academic rules":"Academic Rules","attendance":"Academic Rules","grading":"Academic Rules","ffcs":"Academic Rules",
    "hostel norms":"Hostel Norms","norms":"Hostel Norms",
    "admission process":"Admission Process","process":"Admission Process","procedure":"Admission Process",
    "exam pattern":"Exam Pattern","faq":"FAQ","faqs":"FAQ",
    "contacts":"Contacts","contact":"Contacts","helpline":"Contacts"
}
AUDIENCE_MAP = {"nri":"NRI","foreign":"Foreign","international":"Foreign"}
GENDER_MAP = {"lh":"Female","ladies":"Female","girls":"Female","women":"Female",
              "mh":"Male","mens":"Male","men's":"Male","boys":"Male"}

def build_filters_from_query(q: str) -> Dict[str, Any]:
    ql = q.lower()
    f: Dict[str, Any] = {"campus": "Vellore"}  # default
    for k,v in DOMAIN_MAP.items():
        if k in ql: f["domain"]=v; break
    for k,v in PROGRAM_MAP.items():
        if k in ql: f["program"]=v; break
    for k,v in CATEGORY_MAP.items():
        if k in ql: f["category"]=v; break
    for k,v in AUDIENCE_MAP.items():
        if k in ql: f["audience"]=v; break
    for k,v in GENDER_MAP.items():
        if k in ql: f["gender"]=v; break
    # AY like 2025 or 2025-26
    m = re.search(r"(20\d{2})\D{0,3}(\d{2})", ql)
    if m: f["ay"] = f"{m.group(1)}-{m.group(2)}"
    elif re.search(r"\b(20\d{2})\b", ql):
        yr = re.search(r"\b(20\d{2})\b", ql).group(1)
        f["ay"] = f"{yr}-{str(int(yr[-2:])+1).zfill(2)}"
    return f

def detect_intent(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["hostel","mess","bed","ac","non ac","lh","mh"]):
        return "hostel_fees"
    if any(k in ql for k in ["fee","tuition","category 1","category-1"]):
        return "tuition_fees"
    if any(k in ql for k in ["course","program","branch","offered"]):
        return "courses"
    if any(k in ql for k in ["eligibility","marks","age limit","vitree","viteee","vitmee"]):
        return "eligibility"
    if any(k in ql for k in ["document","affidavit","submission","upload"]):
        return "documents"
    if any(k in ql for k in ["refund","withdrawal","vacate"]):
        return "refund"
    if "faq" in ql or "faqs" in ql:
        return "faq"
    if any(k in ql for k in ["ffcs","attendance","grading","cgpa","regulations"]):
        return "rules"
    return "general"

# -------------------- FAISS load & retrieve -------------------- #
def load_store(index_dir: Path, collection: str, emb) -> FAISS:
    store = FAISS.load_local(
        str(index_dir),
        embeddings=emb,                 # used if you call similarity_search()
        index_name=collection,
        allow_dangerous_deserialization=True
    )
    # we’ll search by vector directly to avoid surprises
    store.embedding_function = emb
    return store

def retrieve(store: FAISS, emb, query: str, k: int = 40):
    vec = emb.embed_query(query)
    docs = store.similarity_search_by_vector(vec, k=k)
    return docs

def postfilter_docs(docs, filt: Dict[str, Any], topn: int = 12):
    if not filt: return docs[:topn]
    def ok(md: Dict[str, Any]):
        for k,v in filt.items():
            if k not in md: continue
            if v and str(md[k]).lower() != str(v).lower(): return False
        return True
    matched = [d for d in docs if ok(d.metadata or {})]
    return (matched or docs)[:topn]

# -------------------- LLM (Gemini) for structured answer -------------------- #
def gemini_answer(context: str, query: str, intent: str) -> Dict[str, Any]:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # generic schema with optional table
    sys_msg = (
        "You are a strict data extractor for VIT Vellore admissions/hostel. "
        "Answer ONLY from the provided context. If a number or value is missing in context, leave it null. "
        "When the intent is fees or hostel_fees, produce a compact table with exact numbers from context. "
        "Do not invent, estimate, or average. Keep currency as shown (INR, USD)."
    )
    schema_hint = {
        "type": "object",
        "properties": {
            "intent": {"type":"string"},
            "audience": {"type":["string","null"]},
            "domain": {"type":["string","null"]},
            "program": {"type":["string","null"]},
            "gender": {"type":["string","null"]},
            "ay": {"type":["string","null"]},
            "currency": {"type":["string","null"]},
            "table": {
                "type":["object","null"],
                "properties": {
                    "title":{"type":["string","null"]},
                    "columns":{"type":"array","items":{"type":"string"}},
                    "rows":{"type":"array","items":{"type":"array","items":{"type":["string","number","null"]}}}
                }
            },
            "bullets": {"type":"array","items":{"type":"string"}},
            "sources": {"type":"array","items":{"type":"string"}}
        },
        "required": ["intent","bullets","sources"]
    }

    prompt = f"""
{sys_msg}

User query: {query}
Detected intent: {intent}

Return STRICT JSON only (no prose) with keys: intent, audience, domain, program, gender, ay, currency, table, bullets, sources.

Context (verbatim chunks):
{context}
"""
    # ask the model for JSON
    resp = model.generate_content(
        [{"role":"user","parts":[prompt]}],
        generation_config={"temperature": 0.1, "max_output_tokens": 1200},
    )
    text = resp.text.strip()
    # some SDK versions wrap code fences; strip them if present
    text = text.removeprefix("```json").removesuffix("```").strip()
    try:
        data = json.loads(text)
    except Exception:
        # fallback minimal answer
        data = {"intent": intent, "bullets": ["(LLM parsing failed. Showing top facts from context.)"], "sources":[]}
    return data

# -------------------- Render to Markdown -------------------- #
def render_markdown(ans: Dict[str, Any]) -> str:
    out = []
    title_bits = []
    if ans.get("domain"): title_bits.append(ans["domain"])
    if ans.get("program"): title_bits.append(ans["program"])
    if ans.get("gender"): title_bits.append(ans["gender"])
    if ans.get("audience"): title_bits.append(ans["audience"])
    if ans.get("ay"): title_bits.append(ans["ay"])
    header = " — ".join(title_bits) if title_bits else None
    if header: out.append(f"### {header}")

    tbl = ans.get("table")
    if isinstance(tbl, dict) and tbl.get("columns") and tbl.get("rows"):
        if tbl.get("title"):
            out.append(f"**{tbl['title']}**")
        # markdown table
        cols = tbl["columns"]
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"]*len(cols)) + "|")
        for row in tbl["rows"]:
            out.append("| " + " | ".join("" if r is None else str(r) for r in row) + " |")
        out.append("")

    bullets = ans.get("bullets") or []
    if bullets:
        out.append("**Highlights:**")
        for b in bullets[:10]:
            out.append(f"- {b}")

    src = list(dict.fromkeys(ans.get("sources") or []))[:5]
    if src:
        out.append("")
        out.append(f"*Sources:* " + ", ".join(src))
    out.append("\n*Note: Information is based on official PDFs you ingested. For the very latest updates, verify on VIT’s website.*")
    return "\n".join(out)

# -------------------- Glue: retrieve → filter → LLM → render -------------------- #
def answer_query(index_dir: Path, collection: str, emb_kind: str, query: str, k: int = 40) -> str:
    emb = GeminiEmbeddings() if emb_kind == "gemini" else STEmbeddings()
    store = load_store(index_dir, collection, emb)

    filters = build_filters_from_query(query)
    intent  = detect_intent(query)

    docs = retrieve(store, emb, query, k=k)
    docs = postfilter_docs(docs, filters, topn=12)

    # pack minimal context with source headers
    ctx_lines = []
    srcs = []
    for d in docs:
        md = d.metadata or {}
        srcs.append(md.get("source_file") or md.get("source_title") or "unknown")
        header = f"[{md.get('source_file','?')}] domain={md.get('domain')} category={md.get('category')} program={md.get('program')} audience={md.get('audience')} gender={md.get('gender')} ay={md.get('ay')}"
        ctx_lines.append(header)
        ctx_lines.append(d.page_content.strip())
        ctx_lines.append("\n")
    context = "\n".join(ctx_lines)[:12000]  # keep prompt smallish

    ans = gemini_answer(context=context, query=query, intent=intent)
    # ensure we preserve sources from our retrieval if LLM omitted them
    if not ans.get("sources"):
        ans["sources"] = list(dict.fromkeys(srcs))[:5]
    return render_markdown(ans)

# -------------------- CLI -------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="Folder containing FAISS files (.faiss/.pkl)")
    ap.add_argument("--collection", default="vit_faq_vellore", help="Index name used when saving")
    ap.add_argument("--emb", choices=["gemini","st"], default="gemini", help="Embedding backend for query")
    ap.add_argument("--q", required=True, help="User query")
    args = ap.parse_args()

    md = answer_query(Path(args.index_dir), args.collection, args.emb, args.q)
    print("\n" + md + "\n")

if __name__ == "__main__":
    main()
