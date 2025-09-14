#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module-1.5: Post-process cleaned chunks to improve quality before embeddings.

Input : data/processed/chunks.jsonl
Output: data/processed/chunks_clean.jsonl
Report: data/processed/clean_report.json
"""

import json, re, argparse, math, statistics, hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SECTION_HINTS = {
    "Fee Structure": [
        "fee structure", "tuition", "fees", "charges", "payment", "category 1", "category 2",
        "non-veg", "special mess", "admission fee", "caution deposit"
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
        "hostel rules", "norms", "biometric", "in/out", "gate", "silence hours", "appliances", "cooking"
    ],
    "Refund Policy": [
        "refund", "withdrawal", "vacate", "deduction", "per month"
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

AY_PAT = re.compile(r'(20\d{2})\D{0,3}(\d{2})')              # 2025-26 / 2025–26
EMAIL_PAT = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_PAT = re.compile(r'(?:(?:\+?\d{1,3}[\s-]?)?\d{10,})')   # simple intl/IN phones
NOTE_STRIP = [
    r'university reserves the right', r'subject to change', r'without prior notice'
]

def sha1(s): return hashlib.sha1(s.encode('utf-8','ignore')).hexdigest()

def refine_category(text, current):
    t = text.lower()
    scores = {}
    for cat, keys in SECTION_HINTS.items():
        scores[cat] = sum(k in t for k in keys)
    best = max(scores, key=scores.get)
    if scores[best] >= 2:               # need at least 2 hits to override
        return best
    return current

def infer_ay(text, current):
    if current: return current
    m = AY_PAT.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return current

def extract_contacts(text):
    emails = list(sorted(set(EMAIL_PAT.findall(text))))
    phones = list(sorted(set(PHONE_PAT.findall(text))))
    return emails, phones

def merge_small_chunks(records, min_len=280, max_len=520):
    """Merge consecutive small chunks within the same (source_file, domain, category, program) group."""
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
            # try to merge
            merged_text = (buf["text"].rstrip() + "\n\n" + rec["text"].lstrip()).strip()
            if len(merged_text) <= max_len:
                buf["text"] = merged_text
                # keep earlier chunk_id; discard rec
                continue
        out.append(buf)
        buf = rec
    if buf is not None:
        out.append(buf)
    return out

def near_dedup(records, threshold=0.97):
    """Remove near-duplicate chunks within the same (source_file, category, program)."""
    if not records: return records
    texts = [r["text"] for r in records]
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    keep = []
    killed = set()
    for i in range(len(records)):
        if i in killed: continue
        keep.append(i)
        # compare with following only
        sims = cosine_similarity(X[i], X[i+1:]).ravel()
        for j, sim in enumerate(sims, start=i+1):
            if sim >= threshold:
                killed.add(j)
    return [records[i] for i in keep], len(records)-len(keep)

def clean_lines(text):
    t = text
    # strip common boilerplate phrases
    for pat in NOTE_STRIP:
        t = re.sub(pat, '', t, flags=re.I)
    # collapse spaces/newlines again
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def load_jsonl(path: Path):
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", dest="out", default="data/processed/chunks_clean.jsonl")
    ap.add_argument("--report", dest="report", default="data/processed/clean_report.json")
    args = ap.parse_args()

    inp = Path(args.inp); outp = Path(args.out); rep = Path(args.report)
    outp.parent.mkdir(parents=True, exist_ok=True)

    recs = load_jsonl(inp)
    before = len(recs)

    # group per file to preserve order and merge small chunks
    groups = defaultdict(list)
    for r in recs:
        groups[r["metadata"].get("source_file")].append(r)
    # sort by chunk_id
    for k in groups:
        groups[k].sort(key=lambda r: r["metadata"].get("chunk_id", 0))

    cleaned, merged_count, refined_cats = [], 0, 0
    inferred_ay = 0
    contacts_added = 0

    for sf, rows in groups.items():
        # pass 1: refine category/ay/contacts, strip boilerplate
        for r in rows:
            r["text"] = clean_lines(r["text"])
            new_cat = refine_category(r["text"], r["metadata"].get("category"))
            if new_cat != r["metadata"].get("category"):
                r["metadata"]["category"] = new_cat
                refined_cats += 1
            new_ay = infer_ay(r["text"], r["metadata"].get("ay"))
            if new_ay and new_ay != r["metadata"].get("ay"):
                r["metadata"]["ay"] = new_ay
                inferred_ay += 1
            emails, phones = extract_contacts(r["text"])
            if emails or phones:
                r["metadata"]["emails"] = emails
                r["metadata"]["phones"] = phones
                contacts_added += 1

        # pass 2: merge small chunks
        merged = merge_small_chunks(rows, min_len=280, max_len=520)
        merged_count += (len(rows) - len(merged))

        # pass 3: near-duplicate removal inside this file
        deduped, killed = near_dedup(merged, threshold=0.97)
        merged_count += killed

        cleaned.extend(deduped)

    # exact dedup across all (by text hash)
    seen = set()
    final = []
    for r in cleaned:
        h = hashlib.sha1(r["text"].encode("utf-8","ignore")).hexdigest()
        if h in seen: 
            continue
        seen.add(h)
        r["metadata"]["content_hash2"] = h
        final.append(r)

    after = len(final)
    report = {
        "input_chunks": before,
        "output_chunks": after,
        "merged_or_removed": before - after,
        "refined_categories": refined_cats,
        "inferred_ay": inferred_ay,
        "contacts_enriched": contacts_added
    }

    with outp.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with rep.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[DONE] Wrote:", outp)
    print("[REPORT]", json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
