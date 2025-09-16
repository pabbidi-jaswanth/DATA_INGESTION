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
    "ug": "UG", "undergrad": "UG",
    "pg": "PG", "postgrad": "PG", "post graduate": "PG",
    "hostel": "Hostel", "accommodation": "Hostel",
    "research": "Research", "phd": "Research", "vitree": "Research",
    "academics": "Academics", "academic": "Academics", "rules": "Academics", "ffcs": "Academics",
    "fees": "Fees", "fee": "Fees", "tuition": "Fees",
    "contacts": "Contacts", "contact": "Contacts", "helpline": "Contacts",
    "faq": "FAQ", "faqs": "FAQ"
}
PROGRAM_MAP = {
    "mca": "MCA", "m.tech": "M.Tech", "mtech": "M.Tech", "msc": "M.Sc", "m.sc": "M.Sc",
    "btech": "B.Tech", "b.tech": "B.Tech", "phd": "Ph.D"
}
CATEGORY_MAP = {
    "refund policy": "Refund Policy", "refund": "Refund Policy", "cancellation": "Refund Policy",
    "fee structure": "Fee Structure", "fees": "Fee Structure", "tuition": "Fee Structure", "hostel fee": "Fee Structure",
    "courses offered": "Courses Offered", "programs offered": "Courses Offered", "programmes offered": "Courses Offered", "courses": "Courses Offered",
    "eligibility criteria": "Eligibility Criteria", "eligibility": "Eligibility Criteria",
    "documents required": "Documents Required", "documents": "Documents Required", "document submission": "Documents Required",
    "academic rules": "Academic Rules", "attendance": "Academic Rules", "grading": "Academic Rules", "ffcs": "Academic Rules",
    "hostel norms": "Hostel Norms", "norms": "Hostel Norms",
    "admission process": "Admission Process", "process": "Admission Process", "procedure": "Admission Process",
    "exam pattern": "Exam Pattern", "faq": "FAQ", "faqs": "FAQ",
    "contacts": "Contacts", "contact": "Contacts", "helpline": "Contacts",
    "payment": "Payment", "pay": "Payment", "transaction": "Payment", "installment": "Payment"
}
AUDIENCE_MAP = {"nri": "NRI", "foreign": "Foreign", "international": "Foreign", "indian": "Indian"}
GENDER_MAP = {
    "lh": "Female", "ladies": "Female", "girls": "Female", "women": "Female",
    "mh": "Male", "mens": "Male", "men's": "Male", "boys": "Male"
}

def build_filters_from_query(q: str) -> Dict[str, Any]:
    ql = q.lower()
    f: Dict[str, Any] = {"campus": "Vellore"}
    for k, v in DOMAIN_MAP.items():
        if k in ql: f["domain"] = v
    for k, v in PROGRAM_MAP.items():
        if k in ql: f["program"] = v
    for k, v in CATEGORY_MAP.items():
        if k in ql: f["category"] = v
    for k, v in AUDIENCE_MAP.items():
        if k in ql: f["audience"] = v
    for k, v in GENDER_MAP.items():
        if k in ql: f["gender"] = v
    m = re.search(r"(20\d{2})\D{0,3}(\d{2})", ql)
    if m: f["ay"] = f"{m.group(1)}-{m.group(2)}"
    elif re.search(r"\b(20\d{2})\b", ql):
        yr = re.search(r"\b(20\d{2})\b", ql).group(1)
        f["ay"] = f"{yr}-{str(int(yr[-2:])+1).zfill(2)}"
    # New: Detect occupancy for hostels
    occ_match = re.search(r"(\d+)[ -]?(sharing|occupancy|seater|bed)", ql)
    if occ_match:
        f["occupancy"] = occ_match.group(1)
    return f

def detect_intent(q: str) -> List[str]:
    ql = q.lower()
    intents = []
    if any(k in ql for k in ["hostel", "mess", "bed", "ac", "non ac", "lh", "mh"]):
        if any(k in ql for k in ["how many", "number of", "count", "blocks", "rooms", "beds"]):
            intents.append("block_counts")
        else:
            intents.append("hostel_fees")
    if any(k in ql for k in ["fee", "tuition", "category 1", "category-1"]):
        intents.append("tuition_fees")
    if any(k in ql for k in ["payment", "pay", "transaction", "installment"]):
        intents.append("payment")
    if any(k in ql for k in ["course", "program", "branch", "offered"]):
        intents.append("courses")
    if any(k in ql for k in ["eligibility", "marks", "age limit", "vitree", "viteee", "vitmee"]):
        intents.append("eligibility")
    if any(k in ql for k in ["document", "affidavit", "submission", "upload"]):
        intents.append("documents")
    if any(k in ql for k in ["refund", "withdrawal", "vacate", "cancellation"]):
        intents.append("refund")
    if any(k in ql for k in ["faq", "faqs"]):
        intents.append("faq")
    if any(k in ql for k in ["ffcs", "attendance", "grading", "cgpa", "regulations", "academics", "academic"]):
        intents.append("academics")
    if any(k in ql for k in ["contacts", "contact", "helpline", "email", "phone"]):
        intents.append("contacts")
    return intents if intents else ["general"]

# ---------------- Load & retrieve ---------------- #
def load_store(index_dir: Path, collection: str, emb: LCEmbeddings) -> FAISS:
    store = FAISS.load_local(
        str(index_dir),
        embeddings=emb,
        index_name=collection,
        allow_dangerous_deserialization=True
    )
    return store

def retrieve(store: FAISS, query: str, k: int = 100):
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

    def is_relevant(md: Dict[str, Any]) -> bool:
        for k, v in filt.items():
            if k in md and v and str(md[k]).lower() != str(v).lower():
                if k in ["gender", "audience", "ay"]:
                    continue
                return False
        return True

    relevant = [d for d in docs if is_relevant(d.metadata or {})]
    if relevant:
        docs = relevant

    want_lh = filt.get("gender", "").lower() == "female"
    want_mh = filt.get("gender", "").lower() == "male"
    want_nri = filt.get("audience", "").lower() == "nri"
    want_foreign = filt.get("audience", "").lower() == "foreign"
    want_indian = filt.get("audience", "").lower() == "indian"
    want_ay2025 = "2025" in (filt.get("ay", "") or "")

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
        if (filt.get("category") or "").lower() in (md.get("category") or "").lower():
            score += 1
        if (filt.get("domain") or "").lower() in (md.get("domain") or "").lower():
            score += 1
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored][:min(topn, 5)]  # Tighter: Max 5 docs

# ---------------- Gemini LLM: force JSON & schemas for various intents ---------------- #
def gemini_answer(context: str, query: str, intent: str) -> Dict[str, Any]:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Schema and hints per intent
    if intent == "block_counts":
        schema = {
            "ac_blocks": 0,
            "non_ac_blocks": 0,
            "ac_rooms": {},
            "non_ac_rooms": {},
            "notes": []
        }
        prompt_hint = "\nExtract ONLY counts for AC/non-AC blocks/rooms. Use query details (e.g., gender, occupancy). No fees or unrelated info."
        table_hint = None
    elif intent == "hostel_fees":
        table_hint = {
            "title": "Hostel Fee Details (VIT Vellore)",
            "columns": ["Hostel/Block", "Occupancy", "AC/Non-AC", "Period", "Fee Amount", "Mess Type", "Mess Fee", "Caution Deposit", "Other", "Total", "Currency", "AY", "Notes"],
            "rows": []
        }
        prompt_hint = "\nFocus on specific query (e.g., boys, AC). Ignore unrelated."
        schema = None
    elif intent == "tuition_fees":
        table_hint = {
            "title": "Tuition Fee Details (VIT Vellore)",
            "columns": ["Program", "Category/Quota", "Year/Sem", "Amount", "Currency", "AY", "Notes"],
            "rows": []
        }
        prompt_hint = "\nExtract exact fees for queried program/domain."
        schema = None
    elif intent == "payment":
        table_hint = {
            "title": "Payment Details (VIT Vellore)",
            "columns": ["Program", "Payment Mode", "Installment Details", "Due Date", "Notes"],
            "rows": []
        }
        prompt_hint = "\nSpecific to payment methods/installments."
        schema = None
    elif intent == "refund":
        table_hint = {
            "title": "Refund Policy (VIT Vellore)",
            "columns": ["Program", "Cancellation Stage", "Refund Amount", "Conditions", "Notes"],
            "rows": []
        }
        prompt_hint = "\nFocus on refund rules for queried area."
        schema = None
    elif intent in ["academics", "eligibility", "documents", "courses"]:
        table_hint = None
        schema = {
            "bullets": [],  # 3-5 key points
            "notes": []
        }
        prompt_hint = "\nProvide 3-5 concise bullets on the topic (e.g., rules, eligibility, courses)."
    elif intent == "contacts":
        table_hint = {
            "title": "Contact Details (VIT Vellore)",
            "columns": ["Department", "Email", "Phone", "Address", "Notes"],
            "rows": []
        }
        prompt_hint = "\nExtract specific contacts (e.g., admissions, hostels)."
        schema = None
    elif intent == "faq":
        table_hint = None
        schema = {
            "faqs": []  # List of {"q": "Question", "a": "Answer"}
        }
        prompt_hint = "\nList 5-10 relevant FAQs as Q&A pairs."
    else:  # general
        table_hint = None
        schema = {
            "bullets": [],
            "notes": []
        }
        prompt_hint = "\nSummarize key info in 3-5 bullets."

    # Trim context
    context = context[:2000] + "\n[Truncated for precision]"

    sys_msg = (
        "You are a precise extractor for VIT Vellore info across all domains (UG/PG/Hostel/Research/Academics/Fees/Contacts/FAQs). Use ONLY provided context.\n"
        "Be specific: Answer exactly the query. No generics or unrelated info.\n"
        f"{prompt_hint}\n"
        "If missing, set to null/0. Return STRICT JSON."
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
        "faqs": [],
        "sources": []
    } if schema is None else schema

    prompt = f"""
{sys_msg}

User query: {query}
Detected intent: {intent}

Return STRICT JSON only (no markdown, no code fences) matching this skeleton:
{json.dumps(json_contract, ensure_ascii=False)}

Rules:
- Extract exactly from context: numbers, currency, details.
- For tables, include only verified rows.
- For bullets/FAQs, keep concise and relevant.
- Include metadata (domain, program, etc.) from context.
- Sources: list of file names.

Context:
{context}
"""

    resp = model.generate_content(
        [{"role": "user", "parts": [prompt]}],
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 1200,
            "response_mime_type": "application/json"
        },
    )
    text = (resp.text or "").strip()
    try:
        parsed = json.loads(text)
        # Post-process for counts/intents
        if intent == "block_counts" and "bullets" not in parsed:
            parsed["bullets"] = [
                f"AC Blocks: {parsed.get('ac_blocks', 0)}",
                f"Non-AC Blocks: {parsed.get('non_ac_blocks', 0)}"
            ]
            for k, v in parsed.get("ac_rooms", {}).items():
                parsed["bullets"].append(f"AC {k}: {v} rooms")
            for k, v in parsed.get("non_ac_rooms", {}).items():
                parsed["bullets"].append(f"Non-AC {k}: {v} rooms")
        elif intent == "faq" and "bullets" not in parsed:
            parsed["bullets"] = [f"Q: {q['q']}\nA: {q['a']}" for q in parsed.get("faqs", [])]
        return parsed
    except Exception as e:
        print(f"[DEBUG] LLM JSON parsing failed: {e}, raw response: {text}")
        return {
            "intent": intent,
            "bullets": ["(LLM parsing failed. Showing top facts from context.)"],
            "sources": []
        }

# ---------------- Render Markdown ---------------- #
def render_markdown(ans: Dict[str, Any]) -> str:
    out = []
    title_bits = []
    for k in ("domain", "program", "gender", "audience", "ay"):
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

    if "ac_blocks" in ans:  # For block_counts
        out.append("**Block Summary:**")
        out.append(f"- **AC Blocks**: {ans['ac_blocks']}")
        out.append(f"- **Non-AC Blocks**: {ans['non_ac_blocks']}")
        if ans.get("ac_rooms"):
            out.append("**AC Room Breakdown:**")
            for k, v in ans["ac_rooms"].items():
                out.append(f"  - {k}: {v} rooms")
        if ans.get("non_ac_rooms"):
            out.append("**Non-AC Room Breakdown:**")
            for k, v in ans["non_ac_rooms"].items():
                out.append(f"  - {k}: {v} rooms")
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
def answer_query(index_dir: Path, collection: str, emb_kind: str, query: str, k: int = 100) -> str:
    emb = GeminiLCEmbeddings() if emb_kind == "gemini" else None
    if emb is None:
        print("[ERROR] Use --emb gemini (local ST not wired here)."); sys.exit(1)
    store = load_store(index_dir, collection, emb)

    filters = build_filters_from_query(query)
    intents = detect_intent(query)

    combined_ans = {"intent": "combined", "table": None, "bullets": [], "sources": []}
    for intent in intents:
        docs = retrieve(store, query, k=20)  # Lower k for less noise
        docs = postfilter_docs(docs, filters, topn=5)

        # Build compact context
        ctx_lines, srcs = [], []
        for d in docs:
            md = d.metadata or {}
            src = md.get("source_file") or md.get("source_title") or "unknown"
            srcs.append(src)
            header = f"[{src}] domain={md.get('domain')} category={md.get('category')} program={md.get('program')} audience={md.get('audience')} gender={md.get('gender')} ay={md.get('ay')}"
            # Snippet for relevance
            content_snip = re.search(r"(?i)(ac|non-ac|block|room|bed|fee|tuition|contact|faq|academic).*?(\d+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", d.page_content, re.DOTALL)
            ctx_lines.append(header)
            ctx_lines.append(content_snip.group(0) if content_snip else d.page_content[:300])
            ctx_lines.append("")
        context = "\n".join(ctx_lines)[:1500]

        # Debug: Log retrieved context
        print(f"[DEBUG] Intent: {intent}, Retrieved context length: {len(context)}")

        ans = gemini_answer(context=context, query=query, intent=intent)
        if ans.get("table") and not combined_ans["table"]:
            combined_ans["table"] = ans["table"]
        combined_ans["bullets"].extend(ans.get("bullets", []))
        combined_ans["sources"].extend(ans.get("sources", []))
        for key in ["audience", "domain", "program", "gender", "ay", "currency"]:
            if ans.get(key):
                combined_ans[key] = ans[key]

    combined_ans["sources"] = list(dict.fromkeys(combined_ans["sources"]))[:5]
    return render_markdown(combined_ans)

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
