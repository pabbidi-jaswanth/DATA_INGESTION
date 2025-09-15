#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG answering for VIT Vellore chatbot (strict JSON, better filters).

Run (PowerShell one-line):
  .\.venv\Scripts\python.exe rag_answer.py --index_dir Data/index/faiss --collection vit_faq_vellore --emb gemini --q "ug nri ladies hostel fees 2025"

Prereqs:
  pip install google-generativeai langchain-community faiss-cpu python-dotenv
Env:
  setx GEMINI_API_KEY "YOUR_KEY_HERE"
"""

import os, sys, json, argparse, re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- Embeddings (LangChain-compatible wrapper) ---------------- #
from langchain_core.embeddings import Embeddings as LCEmbeddings

class GeminiLCEmbeddings(LCEmbeddings):
    """LangChain Embeddings wrapper around Gemini text-embedding-004."""
    def __init__(self, model: str = "models/text-embedding-004", api_key_env: str = "GEMINI_API_KEY"):
        import google.generativeai as genai
        api_key = os.getenv(api_key_env)
        if not api_key:
            print("[ERROR] GEMINI_API_KEY not set.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = model
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            r = self.genai.embed_content(model=self.model, content=t)
            out.append(r["embedding"])
        return out
    def embed_query(self, text: str) -> List[float]:
        r = self.genai.embed_content(model=self.model, content=text)
        return r["embedding"]

# ---------------- Simple query→metadata hints (router) ---------------- #
DOMAIN_MAP = {
    "ug":"UG", "undergrad":"UG",
    "pg":"PG", "postgrad":"PG", "post graduate":"PG",
    "hostel":"Hostel", "accommodation":"Hostel",
    "research":"Research", "phd":"Research", "vitree":"Research",
}
PROGRAM_MAP = {
    "mca":"MCA", "m.tech":"M.Tech", "mtech":"M.Tech", "msc":"M.Sc", "m.sc":"M.Sc",
    "btech":"B.Tech","b.tech":"B.Tech","phd":"Ph.D"
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
AUDIENCE_MAP = {"nri":"NRI","foreign":"Foreign","international":"Foreign","indian":"Indian"}
GENDER_MAP = {"lh":"Female","ladies":"Female","girls":"Female","women":"Female",
              "mh":"Male","mens":"Male","men's":"Male","boys":"Male"}

def build_filters_from_query(q: str) -> Dict[str, Any]:
    ql = q.lower()
    f: Dict[str, Any] = {"campus":"Vellore"}
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
    # academic year in query (2025 or 2025-26)
    m = re.search(r"(20\d{2})\D{0,3}(\d{2})", ql)
    if m: f["ay"] = f"{m.group(1)}-{m.group(2)}"
    elif re.search(r"\b(20\d{2})\b", ql):
        yr = re.search(r"\b(20\d{2})\b", ql).group(1)
        f["ay"] = f"{yr}-{str(int(yr[-2:])+1).zfill(2)}"
    return f

def detect_intent(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["hostel","mess","bed","ac","non ac","lh","mh"]): return "hostel_fees"
    if any(k in ql for k in ["fee","tuition","category 1","category-1"]):     return "tuition_fees"
    if any(k in ql for k in ["course","program","branch","offered"]):         return "courses"
    if any(k in ql for k in ["eligibility","marks","age limit","vitree","viteee","vitmee"]): return "eligibility"
    if any(k in ql for k in ["document","affidavit","submission","upload"]):  return "documents"
    if any(k in ql for k in ["refund","withdrawal","vacate"]):                return "refund"
    if "faq" in ql or "faqs" in ql:                                           return "faq"
    if any(k in ql for k in ["ffcs","attendance","grading","cgpa","regulations"]): return "rules"
    return "general"

# ---------------- Load & retrieve ---------------- #
def load_store(index_dir: Path, collection: str, emb: LCEmbeddings) -> FAISS:
    store = FAISS.load_local(
        str(index_dir),
        embeddings=emb,
        index_name=collection,
        allow_dangerous_deserialization=True
    )
    return store

def retrieve(store: FAISS, query: str, k: int = 40):
    # Let FAISS compute query embeddings via the LCEmbeddings wrapper
    return store.similarity_search(query, k=k)

def _filename_tokens(md: Dict[str, Any]) -> Dict[str, bool]:
    name = (md.get("source_file") or "").lower()
    return {
        "lh": "lh" in name or "ladies" in name or "girls" in name or "women" in name,
        "mh": "mh" in name or "mens" in name or "boys" in name,
        "nri": "nri" in name,
        "foreign": "foreign" in name,
        "indian": "indian" in name,
        "ay2025": "2025-26" in name or "2025_26" in name or "2025–26" in name,
    }

def postfilter_docs(docs, filt: Dict[str, Any], topn: int = 12):
    if not filt:
        return docs[:topn]

    def strict_ok(md: Dict[str, Any]) -> bool:
        # exact match when the key exists
        for k,v in filt.items():
            if k in md and v and str(md[k]).lower() != str(v).lower():
                return False
        return True

    # first pass: strict metadata match
    strict = [d for d in docs if strict_ok(d.metadata or {})]
    if strict:
        docs = strict

    # second pass: filename hints (covers missing audience/gender in hostel PDFs)
    want_lh = filt.get("gender","").lower() == "female"
    want_mh = filt.get("gender","").lower() == "male"
    want_nri = filt.get("audience","").lower() == "nri"
    want_foreign = filt.get("audience","").lower() == "foreign"
    want_indian = filt.get("audience","").lower() == "indian"
    want_ay2025 = "2025" in (filt.get("ay","") or "")

    scored = []
    for d in docs:
        md = d.metadata or {}
        tok = _filename_tokens(md)
        score = 0
        if want_lh and tok["lh"]: score += 2
        if want_mh and tok["mh"]: score += 2
        if want_nri and tok["nri"]: score += 2
        if want_foreign and tok["foreign"]: score += 2
        if want_indian and tok["indian"]: score += 2
        if want_ay2025 and tok["ay2025"]: score += 1
        # category/domain nudges
        if (filt.get("category") or "").lower() == "fee structure" and (md.get("category") or "").lower() == "fee structure":
            score += 1
        if (filt.get("domain") or "").lower() == "hostel" and (md.get("domain") or "").lower() == "hostel":
            score += 1
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [d for s,d in scored]
    return (reranked or docs)[:topn]

# ---------------- Gemini LLM: force JSON & table for fees ---------------- #
def gemini_answer(context: str, query: str, intent: str) -> Dict[str, Any]:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # intent-specific table guidance
    if intent == "hostel_fees":
        table_hint = {
            "title": "Hostel Fee Details (VIT Vellore)",
            "columns": [
                "Hostel/Block","Occupancy","AC/Non-AC","Period",
                "Fee Amount","Mess Type","Mess Fee","Caution Deposit",
                "Other","Total","Currency","AY","Notes"
            ],
            "rows": []
        }
    elif intent == "tuition_fees":
        table_hint = {
            "title": "Tuition Fee Details (VIT Vellore)",
            "columns": ["Program","Category/Quota","Year/Sem","Amount","Currency","AY","Notes"],
            "rows": []
        }
    else:
        table_hint = None

    sys_msg = (
        "You are a precise extractor for VIT Vellore. Use ONLY the provided context.\n"
        "If a value is missing, set it to null. Do not guess.\n"
        "If intent is fees/hostel_fees, return a compact table with exact numbers/currency as shown in context."
    )

    json_contract = {
        "intent": intent,
        "audience": None,
        "domain": None,
        "program": None,
        "gender": None,
        "ay": None,
        "currency": None,
        "table": table_hint,
        "bullets": [],
        "sources": []
    }

    prompt = f"""
{sys_msg}

User query: {query}
Detected intent: {intent}

Return STRICT JSON only (no markdown, no code fences) matching this skeleton:
{json.dumps(json_contract, ensure_ascii=False)}

Rules:
- Numbers and currency must be copied exactly from context.
- For tables, include only rows you can verify from context.
- Keep sources as a list of source file names present in the context headers.

Context:
{context}
"""

    resp = model.generate_content(
        [{"role":"user","parts":[prompt]}],
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 1200,
            "response_mime_type": "application/json"
        },
    )
    text = (resp.text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        # Fallback minimal response if parsing fails
        return {"intent": intent, "bullets": ["(LLM parsing failed. Showing top facts from context.)"], "sources": []}

# ---------------- Render Markdown ---------------- #
def render_markdown(ans: Dict[str, Any]) -> str:
    out = []
    title_bits = []
    for k in ("domain","program","gender","audience","ay"):
        if ans.get(k): title_bits.append(str(ans[k]))
    if title_bits:
        out.append("### " + " — ".join(title_bits))

    tbl = ans.get("table")
    if isinstance(tbl, dict) and tbl.get("columns") and tbl.get("rows"):
        if tbl.get("title"): out.append(f"**{tbl['title']}**")
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
        out.append("*Sources:* " + ", ".join(src))
    out.append("\n*Note: Based on ingested official PDFs. For the very latest updates, verify on VIT’s website.*")
    return "\n".join(out)

# ---------------- Glue: retrieve → filter → LLM → render ---------------- #
def answer_query(index_dir: Path, collection: str, emb_kind: str, query: str, k: int = 50) -> str:
    emb = GeminiLCEmbeddings() if emb_kind == "gemini" else None
    if emb is None:
        print("[ERROR] Use --emb gemini (local ST not wired here)."); sys.exit(1)
    store = load_store(index_dir, collection, emb)

    filters = build_filters_from_query(query)
    intent  = detect_intent(query)

    docs = retrieve(store, query, k=k)
    docs = postfilter_docs(docs, filters, topn=12)

    # build compact context with headers (so sources are visible to LLM)
    ctx_lines, srcs = [], []
    for d in docs:
        md = d.metadata or {}
        src = md.get("source_file") or md.get("source_title") or "unknown"
        srcs.append(src)
        header = f"[{src}] domain={md.get('domain')} category={md.get('category')} program={md.get('program')} audience={md.get('audience')} gender={md.get('gender')} ay={md.get('ay')}"
        ctx_lines.append(header)
        ctx_lines.append(d.page_content.strip())
        ctx_lines.append("")
    context = "\n".join(ctx_lines)[:12000]

    ans = gemini_answer(context=context, query=query, intent=intent)
    if not ans.get("sources"):
        ans["sources"] = list(dict.fromkeys(srcs))[:5]
    return render_markdown(ans)

# ---------------- CLI ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="Folder containing FAISS files")
    ap.add_argument("--collection", default="vit_faq_vellore", help="FAISS index name")
    ap.add_argument("--emb", choices=["gemini"], default="gemini", help="Embedding backend (Gemini)")
    ap.add_argument("--q", required=True, help="User query")
    args = ap.parse_args()

    md = answer_query(Path(args.index_dir), args.collection, args.emb, args.q)
    print("\n" + md + "\n")

if __name__ == "__main__":
    main()
