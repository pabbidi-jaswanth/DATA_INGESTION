#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-pass ingest → clean → sanitize → chunk → dedup → FINAL JSONL
for the VIT Smart FAQ (Vellore MVP).

USAGE:
  pip install "langchain<0.3" pdfplumber python-docx scikit-learn tqdm
  python ingest_clean_finalize.py --in data/raw \
                                  --out data/processed/chunks_clean_final.jsonl \
                                  --catalog data/processed/clean_catalog.csv \
                                  --stats data/processed/clean_stats.txt
"""

import os, re, csv, json, argparse, hashlib, datetime, sys, unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# PDF & DOCX
import pdfplumber
try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Dedup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------ Heuristics ---------------------------------- #

SECTION_HINTS = {
    "Fee Structure": [
        "fee structure", "tuition", "fees", "charges", "payment", "category 1", "category 2",
        "admission fee", "caution deposit", "mess", "special mess", "non-veg", "non veg"
    ],
    "Eligibility Criteria": [
        "eligibility", "minimum marks", "pcm", "pcb", "aggregate", "criteria", "age limit"
    ],
    "Documents Required": [
        "documents required", "affidavit", "undertaking", "certificate", "photograph",
        "marksheet", "reporting", "upload", "submission"
    ],
    "Courses Offered": [
        "programmes offered", "programs offered", "branches", "specialization", "schools", "department"
    ],
    "Academic Rules": [
        "attendance", "grading", "grade point", "ffcs", "regulations", "cgpa"
    ],
    "Hostel Norms": [
        "hostel rules", "norms", "biometric", "in/out", "gate", "appliances", "cooking", "timings"
    ],
    "Refund Policy": [
        "refund", "withdrawal", "vacate", "deduction", "refund policy"
    ],
    "Contacts": [
        "contact", "email", "helpline", "phone", "reach out", "support"
    ],
    "Admission Process": [
        "process", "procedure", "application", "counselling", "slot booking", "steps"
    ],
    "Exam Pattern": [
        "exam pattern", "duration", "sections", "marks", "syllabus", "mcq"
    ],
    "FAQ": [
        "frequently asked questions", "faqs", "question", "answer"
    ],
}

AY_PAT     = re.compile(r'(20\d{2})\D{0,3}(\d{2})')  # 2025-26, 2025–26, 2025_26
EMAIL_PAT  = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_PAT  = re.compile(r'(?:(?:\+?\d{1,3}[\s-]?)?\d{10,})')

PAGE_PATTERNS = [
    r"Page\s+\d+\s*(?:of|/)\s*\d+",
    r"^\s*Vellore\s+Institute\s+of\s+Technology.*$",
]
PAGE_REGEXES = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in PAGE_PATTERNS]

# Replacement map for common Unicode punctuation/whitespace weirdness
REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",   # en/em dash, minus
    "\u00A0": " ",                                  # NBSP
    "\uFFFD": "",                                   # replacement char
}

# Bullets, arrows, dingbats (often from Office/PUA fonts)
ARROW_BULLET_CHARS = [
    "•","●","◦","∙","·","▪","▫","◾","◽","■","□",
    "➤","➔","➜","➝","➞","➟","➡","➠","→","⇒",
    "►","▸","▹","▶","▷","❯","❱","❭",
    "","","","","","","","","","","",
]
ARROW_BULLET_SET = set(ARROW_BULLET_CHARS)


# ------------------------------ Utilities ----------------------------------- #

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def guess_ay_from_name(name: str) -> Optional[str]:
    name = name.replace("_", "-")
    m = AY_PAT.search(name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None

def sanitize_text(s: str) -> str:
    """Unicode normalize + remove PUA/controls + normalize punctuation + map bullets/arrows."""
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    for k, v in REPLACEMENTS.items():
        s = s.replace(k, v)
    # remove (cid:###) artifacts
    s = re.sub(r"\(cid:\d+\)", "", s)
    # map bullets/arrows/checkmarks to "- "
    s = "".join(("- " if ch in ARROW_BULLET_SET else ch) for ch in s)

    # strip other control/private-use chars (keep \n and \t)
    def _ok(ch: str) -> bool:
        cat = unicodedata.category(ch)
        return not (cat and cat[0] == "C") or ch in ("\n", "\t")
    s = "".join(ch for ch in s if _ok(ch))

    # collapse sequences like "--  - -" -> " - "
    s = re.sub(r"(?:\s*-\s*){2,}", " - ", s)
    # tidy whitespace/newlines
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # limit repeated punctuation
    s = re.sub(r'([.!?])\1{2,}', r'\1\1', s)   # "!!!!!" -> "!!"
    s = re.sub(r'(,){2,}', r'\1', s)           # ",,,,"  -> ","
    s = re.sub(r'(-){3,}', r'--', s)           # "-----" -> "--"
    # trim stray punctuation at edges
    s = re.sub(r'^[\s\-\.,;:]+', '', s)
    s = re.sub(r'[\s\-\.,;:]+$', '', s)
    return s.strip()

def clean_text_basic(raw: str) -> str:
    """Remove obvious page artifacts, keep paragraphs but join single line-breaks."""
    if not raw:
        return ""
    txt = raw.replace("\r\n", "\n").replace("\r", "\n")
    for rgx in PAGE_REGEXES:
        txt = rgx.sub("", txt)
    txt = re.sub(r"^\s*\d+\s*$", "", txt, flags=re.MULTILINE)   # standalone page numbers
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)                  # hyphens across lines
    txt = re.sub(r"(?<!\n)\n(?!\n)", " ", txt)                  # join single newlines
    txt = re.sub(r"\n{3,}", "\n\n", txt)                        # keep paragraph breaks
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()

def extract_pdf_text(path: Path) -> str:
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
        return "\n\n".join(pages)
    except Exception as e:
        print(f"[WARN] pdfplumber failed on {path.name}: {e}")
        return ""

def extract_docx_text(path: Path) -> str:
    if not HAS_DOCX:
        return ""
    try:
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        print(f"[WARN] python-docx failed on {path.name}: {e}")
        return ""

def map_metadata_from_path(path: Path) -> Dict[str, Any]:
    n = path.name.lower()
    parts = [p.lower() for p in path.parts]
    meta: Dict[str, Any] = {
        "campus": "Vellore",
        "domain": None,          # UG | PG | Hostel | Research | ...
        "category": None,        # Fee Structure | Documents Required | ...
        "program": None,         # MCA | M.Tech | M.Sc | B.Tech | Ph.D.
        "gender": None,          # Male | Female (hostel)
        "student_year": None,    # First-Year | Senior (hostel)
        "currency": None,        # INR | USD
        "audience": None,        # NRI | Foreign
        "ay": guess_ay_from_name(n),
        "stable_flag": "stable",
    }
    # domain
    if any("hostel" in p for p in parts) or n.startswith(("mh", "lh")):
        meta["domain"] = "Hostel"
    elif any("research" in p for p in parts) or "vitree" in n or "phd" in n or "ph.d" in n:
        meta["domain"] = "Research"
    elif any("pg" == p or "postgrad" in p for p in parts) or any(k in n for k in ["mtech","m.tech","mca","msc","m.sc"]):
        meta["domain"] = "PG"
    elif any("ug" == p for p in parts) or "ug" in n:
        meta["domain"] = "UG"
    # audience
    if "nri" in n and "ug" in n:
        meta["audience"] = "NRI"
    if "foreign" in n and "ug" in n:
        meta["audience"] = "Foreign"
    if "international" in n and "admissions" in n and "fee" in n:
        meta["domain"] = meta["domain"] or "NRI-Intl"
        meta["audience"] = "Foreign"
    # program
    if "mca" in n:
        meta["program"] = "MCA"
    elif "mtech" in n or "m.tech" in n:
        meta["program"] = "M.Tech"
    elif "msc" in n or "m.sc" in n:
        meta["program"] = "M.Sc"
    elif "btech" in n or "b.tech" in n:
        meta["program"] = "B.Tech"
    elif meta["domain"] == "Research":
        meta["program"] = "Ph.D."
    # hostel attrs
    if n.startswith("lh") or "ladies" in n or "girls" in n or "women" in n:
        meta["gender"] = "Female"
    if n.startswith("mh") or "mens" in n or "men's" in n or "boys" in n:
        meta["gender"] = "Male"
    if any(k in n for k in ["first-year","first year","freshers","1st-year"]):
        meta["student_year"] = "First-Year"
    if "senior" in n:
        meta["student_year"] = "Senior"
    # category
    if "refund" in n:
        meta["category"] = "Refund Policy"; meta["stable_flag"] = "volatile"
    elif "affidavit" in n:
        meta["category"] = "Documents Required"
    elif any(k in n for k in ["document","submission","download"]):
        meta["category"] = "Documents Required"
    elif "fee" in n and "hostel" in n:
        meta["category"] = "Fee Structure"; meta["stable_flag"] = "volatile"
    elif "fee" in n:
        meta["category"] = "Fee Structure"; meta["stable_flag"] = "volatile"
    elif "faq" in n:
        meta["category"] = "FAQ"
    elif any(k in n for k in ["process","procedure","admissions process"]):
        meta["category"] = "Admission Process"
    elif any(k in n for k in ["programme","programmes","courses","offered"]):
        meta["category"] = "Courses Offered"
    elif "eligibility" in n:
        meta["category"] = "Eligibility Criteria"
    elif "freshers-hostel-admission-information" in n or ("freshers" in n and "hostel" in n):
        meta["category"] = "Hostel Norms"
    # currency
    if any(k in n for k in ["nri","foreign","international"]):
        meta["currency"] = "USD"
    if meta["category"] == "Fee Structure" and not meta.get("currency"):
        meta["currency"] = "INR"
    return meta

def refine_category_from_text(text: str, current: Optional[str]) -> Optional[str]:
    t = text.lower()
    scores = {cat: sum(kw in t for kw in keys) for cat, keys in SECTION_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else current

def infer_ay_from_text(text: str, current: Optional[str]) -> Optional[str]:
    if current: return current
    m = AY_PAT.search(text)
    return f"{m.group(1)}-{m.group(2)}" if m else current

def extract_contacts(text: str) -> Tuple[List[str], List[str]]:
    emails = sorted(set(EMAIL_PAT.findall(text)))
    phones = sorted(set(PHONE_PAT.findall(text)))
    return emails, phones

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    return splitter.split_text(text)

def merge_small_chunks(records: List[Dict[str, Any]], min_len=280, max_len=520) -> List[Dict[str, Any]]:
    out = []; buf = None
    for rec in records:
        key = (rec["metadata"].get("source_file"),
               rec["metadata"].get("domain"),
               rec["metadata"].get("category"),
               rec["metadata"].get("program"))
        if buf is None:
            buf = rec; continue
        buf_key = (buf["metadata"].get("source_file"),
                   buf["metadata"].get("domain"),
                   buf["metadata"].get("category"),
                   buf["metadata"].get("program"))
        if len(buf["text"]) < min_len and key == buf_key:
            merged_text = (buf["text"].rstrip() + "\n\n" + rec["text"].lstrip()).strip()
            if len(merged_text) <= max_len:
                buf["text"] = merged_text
                continue
        out.append(buf); buf = rec
    if buf is not None:
        out.append(buf)
    return out

def near_dedup(records: List[Dict[str, Any]], threshold=0.97) -> Tuple[List[Dict[str, Any]], int]:
    nrecs = len(records)
    if nrecs < 2:
        return records, 0
    texts = [r["text"] for r in records]
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    if X.shape[1] == 0:
        return records, 0
    keep = []; killed = set(); n = X.shape[0]
    for i in range(n):
        if i in killed: continue
        keep.append(i)
        if i + 1 >= n: continue
        tail = X[i+1:]
        if tail.shape[0] == 0: continue
        sims = cosine_similarity(X[i], tail).ravel()
        for j_off, sim in enumerate(sims, start=1):
            if sim >= threshold:
                killed.add(i + j_off)
    result = [records[idx] for idx in range(n) if idx in keep and idx not in killed]
    return result, len(killed)


# ------------------------------ Main ---------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Ingest + sanitize + chunk + dedup to FINAL JSONL")
    ap.add_argument("--in", dest="in_dir", default="data/raw", help="Input folder (recursive)")
    ap.add_argument("--out", dest="out_jsonl", default="data/processed/chunks_clean_final.jsonl",
                    help="Output FINAL cleaned JSONL")
    ap.add_argument("--catalog", dest="catalog_csv", default="data/processed/clean_catalog.csv",
                    help="Output catalog CSV per-file")
    ap.add_argument("--stats", dest="stats_path", default="", help="Optional stats .txt (glyphs before/after)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    catalog_csv = Path(args.catalog_csv).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    catalog_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning input folder: {in_dir}")
    print(f"[INFO] Writing FINAL chunks : {out_jsonl}")
    print(f"[INFO] Writing catalog      : {catalog_csv}")

    files: List[Path] = []
    for ext in ("*.pdf", "*.PDF", "*.docx", "*.DOCX"):
        files.extend(in_dir.rglob(ext))
    if not files:
        print(f"[ERROR] No PDFs/DOCX found under: {in_dir}"); sys.exit(1)

    file_reports = []
    all_cleaned: List[Dict[str, Any]] = []

    # stats for glyphs before/after
    def has_weird(s: str) -> bool:
        if not s: return False
        return any(ch in s for ch in ARROW_BULLET_SET) or "(cid:" in s or "\uFFFD" in s
    glyphs_before = 0
    glyphs_after  = 0
    total_chunks  = 0

    for path in tqdm(files, desc="Processing files"):
        meta_base = map_metadata_from_path(path)

        # Extract → basic clean → sanitize once globally
        raw = extract_pdf_text(path) if path.suffix.lower() == ".pdf" else extract_docx_text(path)
        cleaned_basic = clean_text_basic(raw)
        cleaned = sanitize_text(cleaned_basic)

        if not cleaned.strip():
            print(f"[WARN] Empty after extract/clean: {path.name}")
            file_reports.append([path.name, 0, 0, 0, "EMPTY"])
            continue

        # Chunk
        chunks = chunk_text(cleaned)
        total_chunks += len(chunks)

        # Build per-chunk records
        recs = []
        mtime = datetime.datetime.utcfromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        for i, ch in enumerate(chunks):
            # extra per-chunk sanitize (idempotent)
            if has_weird(ch): glyphs_before += 1
            ch2 = sanitize_text(ch)
            if has_weird(ch2): glyphs_after += 1
            recs.append({
                "id": f"{path.stem}__{i}",
                "text": ch2,
                "metadata": {
                    **meta_base,
                    "source_file": path.name,
                    "source_path": str(path),
                    "source_title": path.stem,
                    "chunk_id": i,
                    "last_updated": mtime,
                    "content_hash": sha1(ch2),
                }
            })

        # Per-chunk enrich (category/AY/contacts)
        refined_cats = 0; inferred_ays = 0; contacts_enriched = 0
        for r in recs:
            new_cat = refine_category_from_text(r["text"], r["metadata"].get("category"))
            if new_cat != r["metadata"].get("category"):
                r["metadata"]["category"] = new_cat; refined_cats += 1
            new_ay = infer_ay_from_text(r["text"], r["metadata"].get("ay"))
            if new_ay and new_ay != r["metadata"].get("ay"):
                r["metadata"]["ay"] = new_ay; inferred_ays += 1
            emails, phones = extract_contacts(r["text"])
            if emails or phones:
                r["metadata"]["emails"] = emails
                r["metadata"]["phones"] = phones
                contacts_enriched += 1

        # Merge tiny chunks & dedup
        recs.sort(key=lambda x: x["metadata"].get("chunk_id", 0))
        merged = merge_small_chunks(recs, min_len=280, max_len=520)
        if len(merged) >= 2:
            deduped, killed = near_dedup(merged, threshold=0.97)
        else:
            deduped, killed = merged, 0

        # Exact dedup inside file
        seen = set(); unique = []
        for r in deduped:
            h2 = sha1(r["text"])
            if h2 in seen: continue
            seen.add(h2); r["metadata"]["content_hash2"] = h2; unique.append(r)

        all_cleaned.extend(unique)

        file_reports.append([
            path.name, len(chunks), len(merged), len(unique),
            f"refined_cat:{refined_cats}|inferred_ay:{inferred_ays}|contacts:{contacts_enriched}|near_killed:{killed}"
        ])

    # Global exact dedup
    global_seen = set(); final_records: List[Dict[str, Any]] = []
    for r in all_cleaned:
        h = sha1(r["text"])
        if h in global_seen: continue
        global_seen.add(h); final_records.append(r)

    # Write outputs
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with catalog_csv.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["source_file","chunks_raw","chunks_after_merge","chunks_final","notes"])
        for row in file_reports:
            w.writerow(row)

    # Optional stats
    if args.stats_path:
        sp = Path(args.stats_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with sp.open("w", encoding="utf-8") as sf:
            sf.write(f"total_input_chunks: {total_chunks}\n")
            sf.write(f"glyphs_before:      {glyphs_before}\n")
            sf.write(f"glyphs_after:       {glyphs_after}\n")
            sf.write(f"final_chunks:       {len(final_records)}\n")
        print(f"[INFO] wrote stats → {sp}")

    print(f"[DONE] Files processed : {len(files)}")
    print(f"[DONE] Final chunks    : {len(final_records)} → {out_jsonl}")
    print(f"[DONE] Clean catalog   : {catalog_csv}")
    print("[TIP ] Next: use this FINAL JSONL for embeddings & indexing.")

# --------- Optional: simple keyword→filter helper for your router ---------- #
DOMAIN_MAP = {
    "ug":"UG", "undergrad":"UG",
    "pg":"PG", "postgrad":"PG", "post graduate":"PG", "post-graduate":"PG",
    "hostel":"Hostel", "accommodation":"Hostel",
    "research":"Research", "phd":"Research", "vitree":"Research",
}
PROGRAM_MAP = {
    "mca":"MCA", "m.tech":"M.Tech", "mtech":"M.Tech", "msc":"M.Sc", "m.sc":"M.Sc",
    "btech":"B.Tech","b.tech":"B.Tech","phd":"Ph.D"
}
CATEGORY_MAP = {
    "refund policy":"Refund Policy", "refund":"Refund Policy",
    "fee structure":"Fee Structure", "fees":"Fee Structure", "tuition":"Fee Structure",
    "courses offered":"Courses Offered", "programs offered":"Courses Offered", "programmes offered":"Courses Offered",
    "eligibility criteria":"Eligibility Criteria", "eligibility":"Eligibility Criteria",
    "documents required":"Documents Required", "documents":"Documents Required", "document submission":"Documents Required",
    "academic rules":"Academic Rules", "attendance":"Academic Rules", "grading":"Academic Rules", "ffcs":"Academic Rules",
    "hostel norms":"Hostel Norms", "norms":"Hostel Norms",
    "admission process":"Admission Process", "process":"Admission Process", "procedure":"Admission Process",
    "exam pattern":"Exam Pattern", "faq":"FAQ", "faqs":"FAQ",
    "contacts":"Contacts", "contact":"Contacts", "helpline":"Contacts"
}
AUDIENCE_MAP = {
    "nri":"NRI", "foreign":"Foreign", "international":"Foreign"
}
def build_filters_from_query(q: str) -> Dict[str, Any]:
    ql = q.lower()
    where: Dict[str, Any] = {}
    for k,v in DOMAIN_MAP.items():
        if k in ql: where["domain"] = v; break
    for k,v in PROGRAM_MAP.items():
        if k in ql: where["program"] = v; break
    for k,v in CATEGORY_MAP.items():
        if k in ql: where["category"] = v; break
    for k,v in AUDIENCE_MAP.items():
        if k in ql: where["audience"] = v; break
    where.setdefault("campus", "Vellore")   # MVP default
    return where


if __name__ == "__main__":
    main()
