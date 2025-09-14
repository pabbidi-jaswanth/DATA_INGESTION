#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end ingestion & cleaning for VIT Smart FAQ (Vellore MVP)

Features:
- Recursively reads PDFs / DOCX under --in
- Extracts text (pdfplumber / python-docx)
- Cleans page artifacts, joins broken lines, normalizes whitespace
- Infers rich metadata from filenames and folder names
- Heading-aware chunking (≈450 chars, 60 overlap)
- Post-process:
    * merge tiny fragments
    * refine category from section keywords
    * infer AY when missing
    * extract emails/phones into metadata
    * exact + near-duplicate removal (TF-IDF cosine)
- Writes:
    * --out JSONL: chunks ready for embeddings/indexing
    * --catalog CSV: clean per-file summary

Run:
  python ingest_and_clean.py --in data/raw \
                             --out data/processed/chunks_clean.jsonl \
                             --catalog data/processed/clean_catalog.csv
"""

import os, re, csv, json, argparse, hashlib, datetime, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

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


# ------------------------------ Helpers ------------------------------------- #

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

AY_PAT = re.compile(r'(20\d{2})\D{0,3}(\d{2})')  # 2025-26, 2025–26, 2025_26
EMAIL_PAT = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_PAT = re.compile(r'(?:(?:\+?\d{1,3}[\s-]?)?\d{10,})')  # permissive, India+intl

PAGE_PATTERNS = [
    r"Page\s+\d+\s*(?:of|/)\s*\d+",
    r"^\s*Vellore\s+Institute\s+of\s+Technology.*$",
]
PAGE_REGEXES = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in PAGE_PATTERNS]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def guess_ay_from_name(name: str) -> Optional[str]:
    name = name.replace("_", "-")
    m = AY_PAT.search(name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None


def clean_text(raw: str) -> str:
    """Remove common headers/footers, fix hyphenation, join lines, normalize whitespace."""
    if not raw:
        return ""
    txt = raw.replace("\r\n", "\n").replace("\r", "\n")
    for rgx in PAGE_REGEXES:
        txt = rgx.sub("", txt)
    # Remove lines that are just numbers
    txt = re.sub(r"^\s*\d+\s*$", "", txt, flags=re.MULTILINE)
    # Fix hyphenation across line breaks
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
    # Keep paragraph breaks but join single newlines
    txt = re.sub(r"(?<!\n)\n(?!\n)", " ", txt)
    # Collapse 3+ newlines to 2
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    # Collapse repeated spaces/tabs
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()


def extract_pdf_text(path: Path) -> str:
    try:
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                pages.append(t)
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
    """Infer metadata from filename AND folder parts."""
    n = path.name.lower()
    parts = [p.lower() for p in path.parts]

    meta: Dict[str, Any] = {
        "campus": "Vellore",   # MVP fixed
        "domain": None,        # UG | PG | Research | Hostel | NRI-UG | Foreign-UG | NRI-Intl
        "category": None,      # Fee Structure | ... | FAQ | Admission Process
        "program": None,       # B.Tech | M.Tech | MCA | M.Sc | Ph.D. | Direct Ph.D.
        "gender": None,        # Male | Female (hostel)
        "student_year": None,  # First-Year | Senior (hostel)
        "currency": None,      # INR | USD
        "audience": None,      # NRI | Foreign
        "ay": guess_ay_from_name(n),
        "stable_flag": "stable",
    }

    # Domain hints from path
    if any("hostel" in p for p in parts) or n.startswith(("mh", "lh")):
        meta["domain"] = "Hostel"
    elif any("research" in p for p in parts) or "vitree" in n or "phd" in n or "ph.d" in n:
        meta["domain"] = "Research"
    elif any("pg" == p or "postgrad" in p for p in parts) or any(k in n for k in ["mtech", "m.tech", "mca", "msc", "m.sc"]):
        meta["domain"] = "PG"
    elif any("ug" == p for p in parts) or "ug" in n:
        meta["domain"] = "UG"

    # NRI/Foreign audience
    if "nri" in n and "ug" in n:
        meta["domain"] = meta["domain"] or "NRI-UG"
        meta["audience"] = "NRI"
    if "foreign" in n and "ug" in n:
        meta["domain"] = meta["domain"] or "Foreign-UG"
        meta["audience"] = "Foreign"
    if "international" in n and "admissions" in n and "fee" in n:
        meta["domain"] = "NRI-Intl"
        meta["audience"] = "Foreign"

    # Program
    if "mca" in n:
        meta["program"] = "MCA"
    elif "mtech" in n or "m.tech" in n:
        meta["program"] = "M.Tech"
    elif "msc" in n or "m.sc" in n:
        meta["program"] = "M.Sc"
    elif "btech" in n or "b.tech" in n:
        meta["program"] = "B.Tech"
    elif meta["domain"] == "Research":
        meta["program"] = "Ph.D." if "direct" not in n else "Direct Ph.D."

    # Hostel attributes
    if n.startswith("lh") or "ladies" in n or "girls" in n or "women" in n:
        meta["gender"] = "Female"
    if n.startswith("mh") or "mens" in n or "men's" in n or "boys" in n:
        meta["gender"] = "Male"
    if any(k in n for k in ["first-year", "first year", "freshers", "1st-year"]):
        meta["student_year"] = "First-Year"
    if "senior" in n:
        meta["student_year"] = "Senior"

    # Category (filename first)
    if "refund" in n:
        meta["category"] = "Refund Policy"; meta["stable_flag"] = "volatile"
    elif "affidavit" in n:
        meta["category"] = "Documents Required"
    elif any(k in n for k in ["document", "submission", "download"]):
        meta["category"] = "Documents Required"
    elif "fee" in n and "hostel" in n:
        meta["category"] = "Fee Structure"; meta["stable_flag"] = "volatile"
    elif "fee" in n:
        meta["category"] = "Fee Structure"; meta["stable_flag"] = "volatile"
    elif "faq" in n:
        meta["category"] = "FAQ"
    elif any(k in n for k in ["process", "procedure", "admissions process"]):
        meta["category"] = "Admission Process"
    elif any(k in n for k in ["programme", "programmes", "courses", "offered"]):
        meta["category"] = "Courses Offered"
    elif "eligibility" in n:
        meta["category"] = "Eligibility Criteria"
    elif "freshers-hostel-admission-information" in n or ("freshers" in n and "hostel" in n):
        meta["category"] = "Hostel Norms"

    # Currency hints
    if any(k in n for k in ["nri", "foreign", "international"]):
        meta["currency"] = "USD"
    if meta["category"] == "Fee Structure" and not meta.get("currency"):
        meta["currency"] = "INR"

    return meta


def refine_category_from_text(text: str, current: Optional[str]) -> Optional[str]:
    t = text.lower()
    scores = {cat: sum(kw in t for kw in keys) for cat, keys in SECTION_HINTS.items()}
    best = max(scores, key=scores.get)
    # only override if there are enough signals
    if scores[best] >= 2:
        return best
    return current


def infer_ay_from_text(text: str, current: Optional[str]) -> Optional[str]:
    if current:
        return current
    m = AY_PAT.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return current


def extract_contacts(text: str) -> Tuple[List[str], List[str]]:
    emails = list(sorted(set(EMAIL_PAT.findall(text))))
    phones = list(sorted(set(PHONE_PAT.findall(text))))
    return emails, phones


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    return splitter.split_text(text)


def merge_small_chunks(records: List[Dict[str, Any]], min_len=280, max_len=520) -> List[Dict[str, Any]]:
    """Merge consecutive small chunks within same (source_file, domain, category, program)."""
    out = []
    buf = None
    for rec in records:
        key = (rec["metadata"].get("source_file"),
               rec["metadata"].get("domain"),
               rec["metadata"].get("category"),
               rec["metadata"].get("program"))
        if buf is None:
            buf = rec
            continue
        buf_key = (buf["metadata"].get("source_file"),
                   buf["metadata"].get("domain"),
                   buf["metadata"].get("category"),
                   buf["metadata"].get("program"))
        if len(buf["text"]) < min_len and key == buf_key:
            merged_text = (buf["text"].rstrip() + "\n\n" + rec["text"].lstrip()).strip()
            if len(merged_text) <= max_len:
                buf["text"] = merged_text
                continue  # swallow current rec
        out.append(buf)
        buf = rec
    if buf is not None:
        out.append(buf)
    return out


def near_dedup(records: List[Dict[str, Any]], threshold=0.97) -> Tuple[List[Dict[str, Any]], int]:
    """Remove near-duplicate chunks. Safe guards for tiny sets & empty vectors."""
    nrecs = len(records)
    if nrecs < 2:
        return records, 0

    texts = [r["text"] for r in records]
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    if X.shape[1] == 0:  # no features
        return records, 0

    keep = []
    killed = set()
    n = X.shape[0]
    for i in range(n):
        if i in killed:
            continue
        keep.append(i)

        if i + 1 >= n:
            continue
        tail = X[i + 1:]
        if tail.shape[0] == 0:
            continue

        sims = cosine_similarity(X[i], tail).ravel()
        for j_offset, sim in enumerate(sims, start=1):
            if sim >= threshold:
                killed.add(i + j_offset)

    result = [records[idx] for idx in range(n) if idx in keep and idx not in killed]
    return result, len(killed)


# ------------------------------ Main Pipeline ------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Ingest + clean VIT PDFs/DOCX into JSONL chunks")
    ap.add_argument("--in", dest="in_dir", default="data/raw", help="Input folder (recursively scanned)")
    ap.add_argument("--out", dest="out_jsonl", default="data/processed/chunks_clean.jsonl", help="Output cleaned JSONL")
    ap.add_argument("--catalog", dest="catalog_csv", default="data/processed/clean_catalog.csv", help="Output catalog CSV")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    catalog_csv = Path(args.catalog_csv).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    catalog_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning input folder: {in_dir}")
    print(f"[INFO] Writing chunks to    : {out_jsonl}")
    print(f"[INFO] Writing catalog to   : {catalog_csv}")

    # Collect files
    files: List[Path] = []
    for ext in ("*.pdf", "*.PDF", "*.docx", "*.DOCX"):
        files.extend(in_dir.rglob(ext))
    if not files:
        print(f"[ERROR] No PDFs/DOCX found under: {in_dir}")
        sys.exit(1)

    file_reports = []
    all_cleaned: List[Dict[str, Any]] = []

    for path in tqdm(files, desc="Processing files"):
        meta_base = map_metadata_from_path(path)
        # Extract
        if path.suffix.lower() == ".pdf":
            raw = extract_pdf_text(path)
        else:
            raw = extract_docx_text(path)
        cleaned = clean_text(raw)

        if not cleaned.strip():
            print(f"[WARN] Empty after extract/clean: {path.name}")
            file_reports.append([path.name, 0, 0, 0, "EMPTY"])
            continue

        # Chunk
        chunks = chunk_text(cleaned)

        # Build records
        recs = []
        mtime = datetime.datetime.utcfromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        for i, ch in enumerate(chunks):
            recs.append({
                "id": f"{path.stem}__{i}",
                "text": ch,
                "metadata": {
                    **meta_base,
                    "source_file": path.name,
                    "source_path": str(path),
                    "source_title": path.stem,
                    "chunk_id": i,
                    "last_updated": mtime,
                    "content_hash": sha1(ch),
                }
            })

        # Pass 1: refine per-chunk metadata (category/ay/contacts) & tidy lines
        refined_cats = 0
        inferred_ays = 0
        contacts_enriched = 0
        for r in recs:
            t = r["text"].strip()
            # category
            new_cat = refine_category_from_text(t, r["metadata"].get("category"))
            if new_cat != r["metadata"].get("category"):
                r["metadata"]["category"] = new_cat
                refined_cats += 1
            # AY
            new_ay = infer_ay_from_text(t, r["metadata"].get("ay"))
            if new_ay and new_ay != r["metadata"].get("ay"):
                r["metadata"]["ay"] = new_ay
                inferred_ays += 1
            # contacts
            emails, phones = extract_contacts(t)
            if emails or phones:
                r["metadata"]["emails"] = emails
                r["metadata"]["phones"] = phones
                contacts_enriched += 1
            # slight re-trim
            r["text"] = re.sub(r'[ \t]{2,}', ' ', r["text"])
            r["text"] = re.sub(r'\n{3,}', '\n\n', r["text"]).strip()

        # Merge tiny fragments (within same file/domain/category/program)
        # Keep order by chunk_id
        recs.sort(key=lambda x: x["metadata"].get("chunk_id", 0))
        merged = merge_small_chunks(recs, min_len=280, max_len=520)

        # Near-dup removal (safe for small sets)
        if len(merged) >= 2:
            deduped, killed = near_dedup(merged, threshold=0.97)
        else:
            deduped, killed = merged, 0

        # Exact dedup across this file (by text hash2)
        seen = set()
        unique = []
        for r in deduped:
            h2 = sha1(r["text"])
            if h2 in seen:
                continue
            seen.add(h2)
            r["metadata"]["content_hash2"] = h2
            unique.append(r)

        all_cleaned.extend(unique)

        file_reports.append([
            path.name,
            len(chunks),
            len(merged),
            len(unique),
            f"refined_cat:{refined_cats}|inferred_ay:{inferred_ays}|contacts:{contacts_enriched}|near_killed:{killed}"
        ])

    # Exact dedup across ALL files (final pass)
    global_seen = set()
    final_records: List[Dict[str, Any]] = []
    for r in all_cleaned:
        h = sha1(r["text"])
        if h in global_seen:
            continue
        global_seen.add(h)
        final_records.append(r)

    # Write JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write catalog CSV (clean)
    with catalog_csv.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow([
            "source_file", "chunks_raw", "chunks_after_merge",
            "chunks_final", "notes"
        ])
        for row in file_reports:
            w.writerow(row)

    print(f"[DONE] Files processed : {len(files)}")
    print(f"[DONE] Final chunks    : {len(final_records)} → {out_jsonl}")
    print(f"[DONE] Clean catalog   : {catalog_csv}")
    print("[TIP ] Next: feed this JSONL to your embeddings/indexing module.")

if __name__ == "__main__":
    main()
