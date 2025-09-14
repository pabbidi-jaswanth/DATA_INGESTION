#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module-1: Data Ingestion for VIT Smart FAQ (Vellore MVP)

- Input  : data/raw/  (drop all PDFs/DOCX here)
- Output : data/processed/chunks.jsonl  (one JSON per chunk)
           data/processed/catalog.csv   (one row per source file)

Requires: pdfplumber, python-docx, langchain, tqdm
    pip install pdfplumber python-docx langchain tqdm
"""

import os, re, json, csv, argparse, hashlib, datetime, sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# PDF & DOCX extractors
import pdfplumber
try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# Chunking (heading-aware)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


# ------------------------------- Cleaning ---------------------------------- #
PAGE_PATTERNS = [
    r"Page\s+\d+\s*(?:of|/)\s*\d+",
    r"^\s*Vellore\s+Institute\s+of\s+Technology.*$",     # common letterheads (broad)
    r"^\s*L.?&.?T\s*EduTech.*$",
]
PAGE_REGEXES = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in PAGE_PATTERNS]

def clean_text(raw: str) -> str:
    """Light cleanup: remove common headers/footers, fix hyphenation,
    collapse single newlines (keep double as paragraph), normalize spaces."""
    if not raw:
        return ""

    # Normalize newlines
    txt = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Drop frequent page artifacts
    for rgx in PAGE_REGEXES:
        txt = rgx.sub("", txt)

    # Remove lines that are just page numbers or whitespace
    txt = re.sub(r"^\s*\d+\s*$", "", txt, flags=re.MULTILINE)

    # Fix hyphenated line breaks: "admis-\nsion" -> "admission"
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)

    # Keep paragraph breaks but join within paragraphs:
    # replace single \n with space; keep double \n\n
    txt = re.sub(r"(?<!\n)\n(?!\n)", " ", txt)

    # Collapse 3+ newlines to 2
    txt = re.sub(r"\n{3,}", "\n\n", txt)

    # Collapse repeated spaces/tabs
    txt = re.sub(r"[ \t]{2,}", " ", txt)

    # Trim
    txt = txt.strip()
    return txt


# ------------------------------ Metadata ----------------------------------- #
def _guess_ay_from_name(name: str) -> Optional[str]:
    # Try patterns like 2025-26, 2025–26, 25-26, 2025_26
    name = name.replace("_", "-")
    m = re.search(r"(20\d{2})\D{0,3}(\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None

def _bool(val):
    return True if val else False

def map_metadata(filename: str) -> Dict[str, Any]:
    """Infer metadata from filename (lowercased)."""
    n = filename.lower()

    meta: Dict[str, Any] = {
        "campus": "Vellore",       # MVP fixed
        "domain": None,            # UG | PG | Research | Hostel | NRI-UG | Foreign-UG | NRI-Intl
        "category": None,          # Fee Structure | Eligibility | Documents Required | Courses Offered | Academic Rules | Hostel Norms | Refund Policy | Contacts | FAQ | Admission Process | Exam Pattern
        "program": None,           # B.Tech | M.Tech | MCA | M.Sc | Ph.D. | Direct Ph.D.
        "gender": None,            # Male | Female (hostel)
        "student_year": None,      # First-Year | Senior (hostel)
        "currency": None,          # INR | USD
        "audience": None,          # NRI | Foreign (if needed)
        "ay": _guess_ay_from_name(n),
        "stable_flag": "stable",   # fees/refunds/exam windows -> volatile
    }

    # --- Domain ---
    if "vitree" in n or "phd" in n or "ph.d" in n or "research" in n:
        meta["domain"] = "Research"
        if "direct" in n:
            meta["program"] = "Direct Ph.D."
        else:
            meta["program"] = "Ph.D."
    elif n.startswith("mh") or n.startswith("lh") or "hostel" in n:
        meta["domain"] = "Hostel"
    elif "nri" in n and "ug" in n:
        meta["domain"] = "NRI-UG"
        meta["audience"] = "NRI"
    elif "foreign" in n and "ug" in n:
        meta["domain"] = "Foreign-UG"
        meta["audience"] = "Foreign"
    elif "international" in n and "admissions" in n and "fee" in n:
        meta["domain"] = "NRI-Intl"
        meta["audience"] = "Foreign"
    elif "pg" in n or "mtech" in n or "m.tech" in n or "mca" in n or "msc" in n or "m.sc" in n:
        meta["domain"] = "PG"
    elif "ug" in n:
        meta["domain"] = "UG"

    # --- Program (PG/UG specifics) ---
    if "mca" in n:
        meta["program"] = "MCA"
    elif "mtech" in n or "m.tech" in n:
        meta["program"] = "M.Tech"
    elif "msc" in n or "m.sc" in n:
        meta["program"] = "M.Sc"
    elif "btech" in n or "b.tech" in n:
        meta["program"] = "B.Tech"

    # --- Hostel attributes ---
    if n.startswith("lh") or "ladies" in n or "girls" in n or "women" in n:
        meta["gender"] = "Female"
    if n.startswith("mh") or "mens" in n or "men's" in n or "boys" in n:
        meta["gender"] = "Male"
    if "first-year" in n or "first year" in n or "freshers" in n or "1st-year" in n:
        meta["student_year"] = "First-Year"
    if "senior" in n:
        meta["student_year"] = "Senior"

    # --- Category ---
    if "refund" in n:
        meta["category"] = "Refund Policy"
        meta["stable_flag"] = "volatile"
    elif "affidavit" in n:
        meta["category"] = "Documents Required"
    elif "document" in n or "submission" in n or "downloads" in n:
        meta["category"] = "Documents Required"
    elif "fee" in n and "hostel" in n:
        meta["category"] = "Fee Structure"
        meta["stable_flag"] = "volatile"
    elif "fee" in n:
        meta["category"] = "Fee Structure"
        meta["stable_flag"] = "volatile"
    elif "faq" in n:
        meta["category"] = "FAQ"
    elif "process" in n or "admissions process" in n or "procedure" in n:
        meta["category"] = "Admission Process"
    elif "programme" in n or "programmes" in n or "courses" in n or "offered" in n:
        meta["category"] = "Courses Offered"
    elif "eligibility" in n:
        meta["category"] = "Eligibility Criteria"
    elif "freshers-hostel-admission-information" in n or ("freshers" in n and "hostel" in n):
        meta["category"] = "Hostel Norms"

    # --- Currency / Audience Hints ---
    if any(k in n for k in ["nri", "foreign", "international"]):
        meta["currency"] = "USD"
    # Default to INR where hostel/tuition but no USD cue
    if meta["category"] == "Fee Structure" and not meta.get("currency"):
        meta["currency"] = "INR"

    # Exam patterns are usually volatile (VITREE/VITEEE/VITMEE)
    if meta["domain"] == "Research" and meta.get("category") in (None, "Admission Process"):
        # leave category for content-based detection later if needed
        pass

    return meta


# ------------------------------ Extractors --------------------------------- #
def extract_pdf_text(path: Path) -> str:
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = []
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


# ----------------------------- Ingestion Core ------------------------------ #
def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    return splitter.split_text(text)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def ingest_dir(in_dir: Path, out_jsonl: Path, out_catalog: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_catalog.parent.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    for ext in ("*.pdf", "*.PDF", "*.docx", "*.DOCX"):
        files.extend(list(in_dir.rglob(ext)))
    if not files:
        print(f"[ERROR] No PDFs/DOCX found in {in_dir}.")
        sys.exit(1)

    # Catalog headers
    catalog_rows = [["source_file", "size_bytes", "modified_utc", "pages_or_paras",
                     "domain", "category", "program", "gender", "student_year",
                     "audience", "currency", "ay", "stable_flag", "chunk_count"]]

    total_chunks = 0
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for path in tqdm(files, desc="Ingesting"):
            meta = map_metadata(path.name)
            # extract
            if path.suffix.lower() == ".pdf":
                raw = extract_pdf_text(path)
            else:
                raw = extract_docx_text(path)

            cleaned = clean_text(raw)
            if not cleaned.strip():
                print(f"[WARN] Empty after extract/clean: {path.name}")
                continue

            chunks = chunk_text(cleaned)
            mtime = datetime.datetime.utcfromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")

            # write chunks
            for i, ch in enumerate(chunks):
                rec = {
                    "id": f"{path.stem}__{i}",
                    "text": ch,
                    "metadata": {
                        **meta,
                        "source_file": path.name,
                        "source_path": str(path),
                        "source_title": path.stem,
                        "chunk_id": i,
                        "last_updated": mtime,
                        "content_hash": sha1(ch),
                    },
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # catalog row
            pages_or_paras = cleaned.count("\n\n") + 1
            catalog_rows.append([
                path.name,
                path.stat().st_size,
                mtime,
                pages_or_paras,
                meta.get("domain"),
                meta.get("category"),
                meta.get("program"),
                meta.get("gender"),
                meta.get("student_year"),
                meta.get("audience"),
                meta.get("currency"),
                meta.get("ay"),
                meta.get("stable_flag"),
                len(chunks),
            ])

            total_chunks += len(chunks)

    with out_catalog.open("w", newline="", encoding="utf-8") as cf:
        csv.writer(cf).writerows(catalog_rows)

    print(f"[DONE] Wrote {total_chunks} chunks → {out_jsonl}")
    print(f"[DONE] Wrote catalog → {out_catalog}")


# --------------------------------- CLI ------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Ingest VIT PDFs/DOCX → cleaned chunks JSONL")
    p.add_argument("--in", dest="in_dir", default="data/raw", help="Input folder with PDFs/DOCX")
    p.add_argument("--out", dest="out_jsonl", default="data/processed/chunks.jsonl", help="Output JSONL path")
    p.add_argument("--catalog", dest="catalog", default="data/processed/catalog.csv", help="Catalog CSV path")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_jsonl = Path(args.out_jsonl)
    out_catalog = Path(args.catalog)

    ingest_dir(in_dir, out_jsonl, out_catalog)

if __name__ == "__main__":
    main()
