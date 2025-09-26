"""
Router: SQL-first for anything tabular/structured (fees, programs, eligibility, docs, scholarships, hostel),
RAG fallback (FAISS) for long-form “explain/what/how” questions.

Outputs are intentionally crisp: a single table (when applicable) + short bullets.
"""

from app.sql_router import (
    detect_structured_intent, parse_filters,
    sql_hostel_overview, sql_block_contacts, sql_list_blocks,
    sql_programs, sql_eligibility, sql_documents, sql_academic_fees, sql_scholarships
)

# Optional pretty rendering (keep simple to avoid deps on utils/render)
def _fmt_table(tbl: dict) -> str:
    if not tbl or not tbl.get("rows"):
        return "_No matching rows._"
    title  = f"**{tbl.get('title','Results')}**"
    cols   = tbl["columns"]
    header = " | ".join(cols)
    sep    = " | ".join(["---"]*len(cols))
    lines  = [title, "", header, sep]
    for r in tbl["rows"]:
        lines.append(" | ".join(str(x) if x is not None else "" for x in r))
    return "\n".join(lines)

def _fmt_bullets(bullets: list[str]) -> str:
    return "\n".join(f"- {b}" for b in bullets) if bullets else ""

def _pack(table_dict=None, bullets=None):
    parts = []
    if table_dict:
        parts.append(_fmt_table(table_dict))
    if bullets:
        if parts: parts.append("")
        parts.append(_fmt_bullets(bullets))
    return "\n".join(parts) if parts else "_No results._"

def _sql_route(q: str) -> str | None:
    intent = detect_structured_intent(q)
    f = parse_filters(q)

    # HOSTEL intents
    if intent == "contacts":
        res = sql_block_contacts(f)
        return _pack(res.get("table"), res.get("bullets"))

    if intent == "blocks":
        res = sql_list_blocks(f)
        return _pack(res.get("table"), res.get("bullets"))

    if intent in ("hostel", "tabular"):
        res = sql_hostel_overview(f)
        return _pack(res.get("table"), res.get("bullets"))

    # ACADEMICS intents
    if intent == "programs":
        res = sql_programs(f, q)
        return _pack(res.get("table"), res.get("bullets"))
    if intent == "eligibility":
        res = sql_eligibility(f, q)
        return _pack(res.get("table"), res.get("bullets"))
    if intent == "documents":
        res = sql_documents(f, q)
        return _pack(res.get("table"), res.get("bullets"))
    if intent == "fees":
        res = sql_academic_fees(f, q)
        return _pack(res.get("table"), res.get("bullets"))
    if intent == "scholarships":
        res = sql_scholarships(f, q)
        return _pack(res.get("table"), res.get("bullets"))

    return None  # let RAG decide

def answer(query: str) -> str:
    sql_text = _sql_route(query)
    if sql_text is not None:
        return sql_text

    # --------- FAISS fallback (short & source-backed) ----------
    try:
        from app.utils.fallback_rag import answer as rag_fallback
        rag = rag_fallback(query, max_chunks=6, max_tokens=450)  # keep it tight
        return rag or "_I couldn't find that in my sources._"
    except Exception:
        return "_I couldn't find that in my sources._"

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Show MH Senior NRI hostel fees 2025"
    print(answer(q))
