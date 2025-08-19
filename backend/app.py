from __future__ import annotations
import os
import io
import json
import shutil
import uuid
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import fitz  # PyMuPDF
import numpy as np
import faiss
try:
    from sentence_transformers import SentenceTransformer  # optional
    _st_available = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _st_available = False
try:
    from sklearn.feature_extraction.text import HashingVectorizer
    _sk_available = True
except Exception:
    HashingVectorizer = None  # type: ignore
    _sk_available = False
from PIL import Image
import pytesseract
from llm_call import get_llm_response
from generate_audio import generate_audio, generate_audio_podcast
try:
    from dotenv import load_dotenv
    _dotenv_available = True
except Exception:
    load_dotenv = None  # type: ignore
    _dotenv_available = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Load environment from .env files in backend directory (optional)
if _dotenv_available:
    backend_dir = os.path.dirname(__file__)
    # Load .env then .env.local (without override), then fallback to .env.example if needed
    env_main = os.path.join(backend_dir, ".env")
    env_local = os.path.join(backend_dir, ".env.local")
    env_example = os.path.join(backend_dir, ".env.example")
    if os.path.exists(env_main) and load_dotenv is not None:
        load_dotenv(env_main)
    if os.path.exists(env_local) and load_dotenv is not None:
        load_dotenv(env_local)
    # Fallback only if key creds are still missing
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        if os.path.exists(env_example) and load_dotenv is not None:
            load_dotenv(env_example)
STORE_DIR = os.path.join(DATA_DIR, "store")
FILES_LIBRARY_DIR = os.path.join(DATA_DIR, "files", "library")
FILES_CURRENT_DIR = os.path.join(DATA_DIR, "files", "current")
META_PATH = os.path.join(STORE_DIR, "meta.jsonl")
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_HASH_EMBED = os.environ.get("USE_HASH_EMBED", "1") == "1"
HASH_DIM = int(os.environ.get("HASH_DIM", str(2**15)))  # 32768 by default

# Optional Windows tesseract path
if os.environ.get("TESSERACT_PATH"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_PATH"]

os.makedirs(STORE_DIR, exist_ok=True)
os.makedirs(FILES_LIBRARY_DIR, exist_ok=True)
os.makedirs(FILES_CURRENT_DIR, exist_ok=True)

app = FastAPI(title="PDF Selection-to-Search RAG")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=os.path.join(DATA_DIR, "files")), name="files")
os.makedirs(os.path.join(DATA_DIR, "files", "audio"), exist_ok=True)

# Lazy globals
_model: Any = None
_hash_vec: Any = None
_index: Optional[faiss.IndexFlatIP] = None
_dim: Optional[int] = None


def get_model() -> Any:
    global _model, _dim
    if USE_HASH_EMBED:
        # HashingVectorizer doesn't need model; but set dim for FAISS
        global _hash_vec
        if _hash_vec is None:
            if not _sk_available:
                raise RuntimeError("scikit-learn not installed; install scikit-learn or set USE_HASH_EMBED=0 with sentence-transformers installed.")
            # Use unigrams+bigrams and English stop words to improve retrieval quality
            _hash_vec = HashingVectorizer(
                n_features=HASH_DIM,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
                stop_words="english",
            )  # type: ignore[operator]
            _dim = HASH_DIM
        return None  # type: ignore
    if _model is None:
        if not _st_available:
            raise RuntimeError("sentence-transformers not installed. Leave USE_HASH_EMBED=1 (default) or install optional requirements-sbert.txt.")
        _model = SentenceTransformer(MODEL_NAME)  # type: ignore[operator]
        # warmup to set dim
        v = _model.encode(["warmup"], normalize_embeddings=True)
        _dim = int(v.shape[1])
    return _model


def get_index() -> faiss.IndexFlatIP:
    global _index, _dim
    if _index is None:
        # Try load
        if os.path.exists(INDEX_PATH):
            _index = faiss.read_index(INDEX_PATH)
            if _index is not None:
                _dim = _index.d
        else:
            if _dim is None:
                get_model()
            _index = faiss.IndexFlatIP(_dim)
    return _index


def save_index():
    global _index
    if _index is not None:
        faiss.write_index(_index, INDEX_PATH)


def add_meta(rows: List[Dict[str, Any]]):
    with open(META_PATH, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_meta() -> List[Dict[str, Any]]:
    if not os.path.exists(META_PATH):
        return []
    rows = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def display_name_for_file(fn: str) -> str:
    return fn.split('_', 1)[1] if '_' in fn else fn


# --------- PDF parsing, OCR, chunking ---------

def extract_text_with_ocr(doc_path: str) -> List[Dict[str, Any]]:
    """
    Return list of pages: {page_num, text, outline_hint}
    OCR used when PyMuPDF text is too sparse.
    """
    pages: List[Dict[str, Any]] = []
    doc = fitz.open(doc_path)

    # outline map: page -> headings list
    outlines: Dict[int, List[str]] = {}
    try:
        for toc_item in doc.get_toc(simple=True):
            # toc_item: [level, title, page]
            _, title, page = toc_item
            outlines.setdefault(page - 1, []).append(title)
    except Exception:
        pass

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) < 20:  # likely scanned
            pix = page.get_pixmap(dpi=200, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="eng").strip()
        pages.append({
            "page_num": i + 1,
            "text": text,
            "outline_hint": "; ".join(outlines.get(i, [])) if outlines.get(i) else None,
        })
    doc.close()
    return pages


def heuristic_headings(lines: List[str]) -> List[str]:
    """
    Naive heading detection: treat short Title Case or ALL CAPS lines as headings.
    Returns a list the same length as lines with the most recent detected heading
    propagated, or None if none seen yet.
    """
    heads: List[Optional[str]] = []
    last_head: Optional[str] = None
    for ln in lines:
        s = ln.strip()
        if not s:
            heads.append(last_head)
            continue
        words = s.split()
        is_title = s.istitle() or (s[:1].isupper() and any(c.islower() for c in s[1:]))
        is_all_caps = s.isupper()
        if (is_title or is_all_caps) and len(words) <= 12 and len(s) <= 120:
            last_head = s
        heads.append(last_head)
    # type: ignore[return-value] — allow Optional[str]s; consumers guard appropriately
    return heads  # type: ignore[return-value]


def chunk_text(page_text: str, page_num: int, file_name: str, outline_hint: Optional[str], chunk_size=1000, overlap=200) -> List[Dict[str, Any]]:
    """Split a page's text into overlapping chunks, attaching a best-guess heading."""
    # Split to lines for heading heuristics
    lines = page_text.splitlines()
    heads_seq = heuristic_headings(lines)
    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(page_text):
        end = min(len(page_text), i + chunk_size)
        chunk = page_text[i:end]
        # find a heading for the chunk from the lines covered roughly by i..end
        # crude map: compute line index by cumulative char counts
        cum = 0
        cand_head: Optional[str] = None
        for ln, h in zip(lines, heads_seq):
            cum += len(ln) + 1
            if cum >= i:
                cand_head = h
                break
        chunks.append({
            "text": chunk,
            "page": page_num,
            "file": file_name,
            "heading": cand_head or outline_hint,
        })
        if end == len(page_text):
            break
        i = max(0, end - overlap)
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    if USE_HASH_EMBED:
        # HashingVectorizer returns csr sparse; convert to dense float32 l2-normalized
        global _hash_vec
        if _hash_vec is None:
            get_model()
        X = _hash_vec.transform(texts)
        # l2 normalization is handled by vectorizer; ensure dense float32
        vecs = X.toarray().astype("float32")
        return vecs
    else:
        model = get_model()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype("float32")


# --------- API Endpoints ---------

@app.post("/api/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    """Bulk ingest PDFs as library documents."""
    saved_paths = []
    for uf in files:
        fname = f"{uuid.uuid4().hex}_{uf.filename}"
        dest = os.path.join(FILES_LIBRARY_DIR, fname)
        with open(dest, "wb") as out:
            shutil.copyfileobj(uf.file, out)
        saved_paths.append((fname, dest))

    # parse + chunk
    all_chunks: List[Dict[str, Any]] = []
    for fname, path in saved_paths:
        pages = extract_text_with_ocr(path)
        for p in pages:
            chunks = chunk_text(p["text"], p["page_num"], fname, p["outline_hint"]) \
                if p["text"] else []
            all_chunks.extend(chunks)

    if not all_chunks:
        return JSONResponse({"added": 0})

    # embed + add to faiss
    texts = [c["text"] for c in all_chunks]
    vecs = embed_texts(texts)
    index = get_index()
    index.add(vecs)
    save_index()

    # persist meta rows, track offset ids
    start_id = index.ntotal - vecs.shape[0]
    rows = []
    for off, c in enumerate(all_chunks):
        rows.append({
            "id": start_id + off,
            "file": c["file"],
            "page": c["page"],
            "heading": c.get("heading"),
            "text": c["text"][:1000],  # cap stored snippet
        })
    add_meta(rows)

    return {"added": len(all_chunks)}


@app.post("/api/current")
async def set_current_pdf(file: UploadFile = File(...)):
    """Sets the current PDF and returns a URL to load in viewer."""
    # Clear previous current files
    for f in os.listdir(FILES_CURRENT_DIR):
        try:
            os.remove(os.path.join(FILES_CURRENT_DIR, f))
        except Exception:
            pass
    fname = f"current_{uuid.uuid4().hex}_{file.filename}"
    dest = os.path.join(FILES_CURRENT_DIR, fname)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    url = f"/files/current/{fname}"
    return {"url": url, "fileName": file.filename}


@app.post("/api/search")
async def search_similar(query: str = Form(...), k: int = Form(5)):
    meta = load_meta()
    if not meta:
        return {"results": []}
    index = get_index()
    qv = embed_texts([query])
    D, I = index.search(qv, min(k, index.ntotal))
    hits = []
    meta_by_id = {row["id"]: row for row in meta}
    for idx, score in zip(I[0].tolist(), D[0].tolist()):
        row = meta_by_id.get(idx)
        if not row:
            continue
        hits.append({
            "score": float(score),
            "file": row["file"],
            "page": row["page"],
            "heading": row.get("heading"),
            "text": row.get("text"),
            "fileUrl": f"/files/library/{row['file']}",
        })
    return {"results": hits}


def search_with_threshold(query: str, top_k: int = 20, threshold: float = 0.2) -> List[Dict[str, Any]]:
    meta = load_meta()
    if not meta:
        return []
    index = get_index()
    qv = embed_texts([query])
    # Get more candidates then filter
    k = min(top_k, index.ntotal)
    D, I = index.search(qv, k)
    meta_by_id = {row["id"]: row for row in meta}
    hits: List[Dict[str, Any]] = []
    for idx, score in zip(I[0].tolist(), D[0].tolist()):
        if idx < 0:
            continue
        if float(score) < threshold:
            # As inner product with l2 normalized vectors, scores are cosine-sim in [-1,1]
            # Stop early if sorted descending and below threshold
            continue
        row = meta_by_id.get(idx)
        if not row:
            continue
        hits.append({
            "score": float(score),
            "id": idx,
            "file": row["file"],
            "page": row["page"],
            "heading": row.get("heading"),
            "text": row.get("text"),
            "fileUrl": f"/files/library/{row['file']}",
        })
    # Sort desc by score
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits


# ---------- Enhanced retrieval utilities (CPU-only) ----------
def _normalize_text(t: str) -> str:
    return " ".join((t or "").split())


def _rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion.
    rank_lists: list of lists of doc ids ordered best->worst.
    Returns: id -> fused score.
    """
    rrf_scores: Dict[int, float] = {}
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks):
            # 1 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + r + 1)
    return rrf_scores


def search_multi_rrf(
    queries: List[str],
    top_k: int = 20,
    threshold: float = 0.2,
) -> List[Dict[str, Any]]:
    """Run multiple query variants and fuse with RRF; filter by cosine threshold.
    Output shape matches search_with_threshold.
    """
    meta = load_meta()
    if not meta:
        return []
    index = get_index()
    if index.ntotal <= 0:
        return []
    meta_by_id = {row["id"]: row for row in meta}

    # Prepare embeddings for each query variant
    q_texts = [_normalize_text(q) for q in queries if _normalize_text(q)]
    if not q_texts:
        return []
    # Search with a larger candidate pool to allow fusion
    cand_k = min(max(top_k * 3, 50), index.ntotal)
    all_rank_lists: List[List[int]] = []
    best_cos_by_id: Dict[int, float] = {}
    for q in q_texts:
        qv = embed_texts([q])
        D, I = index.search(qv, cand_k)
        idxs = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0:
                continue
            idxs.append(idx)
            # track best cosine per id
            if (idx not in best_cos_by_id) or (float(score) > best_cos_by_id[idx]):
                best_cos_by_id[idx] = float(score)
        all_rank_lists.append(idxs)

    if not all_rank_lists:
        return []

    fused = _rrf_fuse(all_rank_lists, k=60)
    # Build candidates that pass threshold by best cosine across variants
    cand_ids = [doc_id for doc_id, _ in fused.items() if best_cos_by_id.get(doc_id, -1.0) >= threshold]
    # Sort by fused score then by best cosine as tiebreaker
    cand_ids.sort(key=lambda i: (fused.get(i, 0.0), best_cos_by_id.get(i, -1.0)), reverse=True)

    hits: List[Dict[str, Any]] = []
    for doc_id in cand_ids[:top_k]:
        row = meta_by_id.get(doc_id)
        if not row:
            continue
        hits.append({
            "score": float(best_cos_by_id.get(doc_id, 0.0)),  # keep cosine in score field
            "id": doc_id,
            "file": row["file"],
            "page": row["page"],
            "heading": row.get("heading"),
            "text": row.get("text"),
            "fileUrl": f"/files/library/{row['file']}",
            "_fused": fused.get(doc_id, 0.0),  # internal tie-breaker, not used in responses
        })
    return hits


def build_section_for_hit(primary: Dict[str, Any], meta: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Given a primary hit and full meta list, assemble the 'section' text by grouping
    all chunks from the same (file, heading). If heading is None, fall back to (file, page).
    Returns a section dict and the list of rows used.
    """
    file = primary.get("file")
    heading = primary.get("heading")
    page = primary.get("page")
    if heading:
        rows = [r for r in meta if r.get("file") == file and r.get("heading") == heading]
    else:
        # use same page only; could optionally include neighbors
        rows = [r for r in meta if r.get("file") == file and r.get("page") == page]
    # sort by page then id
    rows.sort(key=lambda r: (r.get("page", 0), r.get("id", 0)))
    text = "\n\n".join((r.get("text") or "") for r in rows if r.get("text"))
    page_vals: List[int] = []
    for r in rows:
        p = r.get("page")
        if isinstance(p, int):
            page_vals.append(p)
    pages = sorted(set(page_vals))
    sec = {
        "file": file,
        "fileUrl": f"/files/library/{file}",
        "displayName": display_name_for_file(file or ""),
        "heading": heading,
        "pages": pages,
        "text": text,
    }
    return sec, rows


def _token_overlap(a: str, b: str) -> float:
    import re
    A = set(re.findall(r"[A-Za-z0-9_]+", (a or "").lower()))
    B = set(re.findall(r"[A-Za-z0-9_]+", (b or "").lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A))


def _name_overlap(selection: str, file: Optional[str], display: Optional[str], heading: Optional[str]) -> float:
    """Lightweight lexical overlap against filename/display/heading to help anchor disambiguation."""
    names = []
    if file:
        names.append(file)
    if display:
        names.append(display)
    if heading:
        names.append(heading)
    text = " \n".join(names)
    return _token_overlap(selection, text)

def _reference_penalty(heading: Optional[str], text: str) -> float:
    """Heuristic penalty for reference-like sections to avoid selecting citations as anchors.
    Returns a value in [0, 1.5] that can be scaled in scoring.
    """
    try:
        h = (heading or "").lower()
        pen = 0.0
        if any(k in h for k in ["reference", "bibliograph", "works cited"]):
            pen += 1.0
        tl = (text or "")
        import re
        # Count bracketed numeric citations like [12], [3,4]
        bracket_cites = len(re.findall(r"\[[0-9,\-\s]{1,}\]", tl))
        if bracket_cites >= 2:
            pen += 0.5
        tl_low = tl.lower()
        if ("doi" in tl_low) or ("arxiv" in tl_low):
            pen += 0.3
        return min(1.5, pen)
    except Exception:
        return 0.0

def _extract_acronyms_and_keywords(selection: str) -> List[str]:
    """Extract useful acronyms like ViT, DDPM, GAN and key Title-Case phrases from the selection.
    Returns a deduped short list to be used as extra queries for anchoring/related.
    """
    import re
    sel = (selection or "").strip()
    out: List[str] = []
    seen = set()
    if not sel:
        return out
    # Acronyms within parentheses: Vision Transformer (ViT) -> ViT
    for m in re.finditer(r"\(([A-Z][A-Za-z0-9]{1,9})\)", sel):
        token = m.group(1).strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    # Standalone all-caps or CamelCase short tokens (2-6 chars): ViT, DDPM, GAN, BERT
    for m in re.finditer(r"\b([A-Z]{2,6}|[A-Z][a-zA-Z]{1,5}[A-Z][A-Za-z]{0,3})\b", sel):
        token = m.group(1).strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    # Title-case bigrams: Vision Transformer, Residual Network (limit length)
    for m in re.finditer(r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b", sel):
        phrase = m.group(1).strip()
        if 3 <= len(phrase) <= 40 and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    # Keep it small
    return out[:6]

def _strip_citation_noise(text: str) -> str:
    """Remove common citation artifacts like [12], [3,4], 'doi', 'arXiv' and URLs to reduce reference bias."""
    import re
    t = (text or "")
    # Remove bracket citations
    t = re.sub(r"\[[0-9,\-\s]{1,}\]", " ", t)
    # Remove DOI/arXiv mentions
    t = re.sub(r"\bdoi:\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\barxiv:\S+", " ", t, flags=re.IGNORECASE)
    # Remove raw URLs
    t = re.sub(r"https?://\S+", " ", t)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


@app.post("/api/insights")
async def generate_insights(
    selection: str = Form(...),
    top_k: int = Form(20),
    threshold: float = Form(0.2),
    current_url: Optional[str] = Form(None),
    stream: bool = Form(False)
):
    """
    Generate grounded insights for a selected text:
    1) Find the best-matching chunk in the index
    2) Expand to the whole section (same file+heading, or same file+page)
    3) Retrieve additional relevant sections above a similarity threshold
    4) Send the selection + sections to the configured LLM for analysis
    """
    sel = (selection or "").strip()
    if not sel:
        return JSONResponse(status_code=400, content={"error": "selection is empty"})

    meta = load_meta()
    if not meta:
        return {"analysis": None, "selection": sel, "home": None, "related": [], "provider": os.getenv("LLM_PROVIDER", "gemini")}

    # Build home section from the best hit using the selection to find the anchor
    # First, find candidates with the raw selection to locate the anchor chunk
    # Anchor search with adaptive threshold fallback
    # Add acronym/keyword expansions for more precise anchoring (e.g., ViT, DDPM)
    _ak = _extract_acronyms_and_keywords(sel)
    anchor_queries = [sel] + _ak if _ak else [sel]
    anchor_hits = search_multi_rrf(anchor_queries, top_k=max(10, top_k), threshold=max(0.0, threshold - 0.05)) or []
    if not anchor_hits and threshold > 0.0:
        anchor_hits = search_multi_rrf(anchor_queries, top_k=max(10, top_k), threshold=max(0.0, threshold - 0.1)) or []
    if not anchor_hits and threshold > 0.0:
        anchor_hits = search_multi_rrf(anchor_queries, top_k=max(10, top_k), threshold=0.0) or []
    if not anchor_hits:
        return {"analysis": None, "selection": sel, "home": None, "related": [], "provider": os.getenv("LLM_PROVIDER", "gemini")}
    # Rerank top few anchor candidates by lexical overlap with built section text
    best_primary = None
    best_score = -1.0
    for cand in anchor_hits[:10]:
        sec_tmp, _ = build_section_for_hit(cand, meta)
        ov = _token_overlap(sel, sec_tmp.get("text", ""))
        name_ov = _name_overlap(sel, cand.get("file"), sec_tmp.get("displayName"), sec_tmp.get("heading"))
        ref_pen = _reference_penalty(sec_tmp.get("heading"), sec_tmp.get("text") or "")
        # Emphasize name overlap; demote reference-like sections slightly
        s = 0.8 * float(cand.get("score", 0.0)) + 0.10 * ov + 0.10 * name_ov - 0.12 * ref_pen
        if s > best_score:
            best_score = s
            best_primary = cand
    primary = best_primary or anchor_hits[0]
    home_section, home_rows = build_section_for_hit(primary, meta)

    # Extract the paragraph within the home section that contains the selection (case-insensitive)
    def extract_paragraph(sel_text: str, sec_text: str) -> str:
        s = (sel_text or "").strip()
        t = (sec_text or "").strip()
        if not s or not t:
            return s or t
        low_s = s.lower()
        # split by paragraph boundaries (double newline) first
        paras = [p.strip() for p in t.split("\n\n") if p.strip()]
        for p in paras:
            if low_s in p.lower():
                return p
        # fallback: sentence-ish split
        import re
        sentences = re.split(r"(?<=[.!?])\s+", t)
        for p in sentences:
            if low_s in p.lower():
                return p
        return s

    selected_para = extract_paragraph(sel, home_section.get("text", ""))

    # Use the entire home section text as the semantic query to retrieve related sections
    # Build multiple query variants for better recall
    heading_hint = home_section.get("heading") or ""
    # Sanitize query text to avoid citation noise
    query_text = _strip_citation_noise(home_section.get("text") or selected_para or sel)
    q_variants = [
        query_text,
        sel,
        (selected_para or sel),
        (heading_hint + ": " + sel) if heading_hint else sel,
    ]
    # Add acronym/keyword expansions as additional variants
    if _ak:
        q_variants.extend(_ak[:4])
    hits = search_multi_rrf(q_variants, top_k=top_k, threshold=threshold)
    if not hits and threshold > 0.0:
        hits = search_multi_rrf(q_variants, top_k=top_k, threshold=max(0.0, threshold - 0.05))
    if not hits and threshold > 0.0:
        hits = search_multi_rrf(q_variants, top_k=top_k, threshold=0.0)
    if not hits:
        return {"analysis": None, "selection": sel, "home": home_section, "related": [], "provider": os.getenv("LLM_PROVIDER", "gemini")}

    home_cite = {
        "label": "Home",
        "file": home_section["file"],
        "displayName": home_section["displayName"],
        "fileUrl": home_section["fileUrl"],
        "heading": home_section.get("heading"),
        "pages": home_section.get("pages"),
    }

    # Group other hits into sections and pick top few distinct sections
    # Exclude home section matches
    def sec_key(h: Dict[str, Any]) -> Tuple[str, str]:
        file_key = str(h.get("file") or "")
        sec = h.get("heading")
        if not sec:
            sec = f"PAGE-{h.get('page')}"
        return (file_key, str(sec))

    home_key = sec_key(primary)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for h in hits[1:]:
        k = sec_key(h)
        if k == home_key:
            continue
        grouped.setdefault(k, []).append(h)

    # Score per section = max score among its hits
    # Section-level scoring: blend of max score and average of top-2; boost same-file slightly for cohesion
    def section_score(section_hits: List[Dict[str, Any]]) -> float:
        scores = sorted([h["score"] for h in section_hits], reverse=True)
        mx = scores[0] if scores else 0.0
        avg_top2 = sum(scores[:2]) / min(2, len(scores)) if scores else 0.0
        return 0.7 * mx + 0.3 * avg_top2

    ranked_sections = []
    for key, hs in grouped.items():
        s = section_score(hs)
        # slight same-file boost
        if key[0] == home_section.get("file"):
            s += 0.02
        # Demote reference-like sections
        tmp_primary = {"file": key[0], "heading": None, "page": None}
        if key[1].startswith("PAGE-"):
            try:
                tmp_primary["page"] = int(key[1].split("-", 1)[1])
            except Exception:
                tmp_primary["page"] = None
        else:
            tmp_primary["heading"] = key[1]
        sec_tmp, _ = build_section_for_hit(tmp_primary, meta)
        ref_pen = _reference_penalty(sec_tmp.get("heading"), sec_tmp.get("text") or "")
        s = s - 0.10 * ref_pen
        ranked_sections.append((key, hs, s))
    ranked_sections.sort(key=lambda x: x[2], reverse=True)

    related_sections: List[Dict[str, Any]] = []
    char_budget = 4000  # tighter budget
    used_chars = len(home_section.get("text", ""))
    # Diversity: avoid picking multiple sections with highly overlapping text or same heading repeatedly
    seen_keys: set = set()
    seen_text_fingerprints: set = set()
    # Add home text fingerprint to avoid duplicating it in related
    try:
        import hashlib as _hashlib
        _home_fp = _hashlib.md5((home_section.get("text") or "").strip().encode("utf-8")).hexdigest()
        seen_text_fingerprints.add(_home_fp)
    except Exception:
        pass
    # Track per-file counts to encourage diversity (limit 1 per file)
    file_pick_counts: Dict[str, int] = {}
    for (file, heading_or_page), _hits, _score in ranked_sections:
        # Build that section from meta
        tmp_primary = {"file": file, "heading": None, "page": None}
        if heading_or_page.startswith("PAGE-"):
            try:
                tmp_primary["page"] = int(heading_or_page.split("-", 1)[1])
            except Exception:
                tmp_primary["page"] = None
        else:
            tmp_primary["heading"] = heading_or_page
        if (file, heading_or_page) in seen_keys:
            continue
        sec, rows = build_section_for_hit(tmp_primary, meta)
        if not sec.get("text"):
            continue
        # Skip if near-duplicate by simple fingerprint
        import hashlib
        fp = hashlib.md5((sec.get("text") or "").strip().encode("utf-8")).hexdigest()
        if fp in seen_text_fingerprints:
            continue
        seen_keys.add((file, heading_or_page))
        seen_text_fingerprints.add(fp)
        # Enforce per-file cap (at most 1 related per file)
        fkey = sec.get("file") or ""
        if fkey:
            if file_pick_counts.get(fkey, 0) >= 1:
                continue
        # Limit per-section length
        sec_text = sec["text"]
        if len(sec_text) > 1200:
            sec_text = sec_text[:1200]
        # Check budget
        if used_chars + len(sec_text) > char_budget:
            break
        used_chars += len(sec_text)
        related_sections.append({
            "file": sec["file"],
            "displayName": sec["displayName"],
            "fileUrl": sec["fileUrl"],
            "heading": sec.get("heading"),
            "pages": sec.get("pages"),
            "text": sec_text,
        })
        if fkey:
            file_pick_counts[fkey] = file_pick_counts.get(fkey, 0) + 1
        if len(related_sections) >= 3:
            break

    # Prepare LLM messages
    provider = os.getenv("LLM_PROVIDER", "gemini")
    frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    backend_origin = os.getenv("BACKEND_ORIGIN", "http://localhost:8000")

    # Helper to build a viewer deep link
    from urllib.parse import quote
    def viewer_link(file_url: str, pages: Optional[List[int]], q: Optional[str]) -> str:
        url = f"{backend_origin}{file_url}"
        page = pages[0] if pages and isinstance(pages[0], int) else None
        qp = f"&page={page}" if page else ""
        qq = f"&q={quote((q or '')[:200])}" if q else ""
        return f"{frontend_origin}/viewer?url={quote(url)}{qp}{qq}"

    # Attach links for citations
    home_section["link"] = viewer_link(home_section.get("fileUrl", ""), home_section.get("pages"), selected_para or sel)
    for rs in related_sections:
        rs["link"] = viewer_link(rs.get("fileUrl", ""), rs.get("pages"), rs.get("text"))

    # Identify "future" related sections in the same document (later than the selected location)
    future_sections: List[Dict[str, Any]] = []
    try:
        home_max_page = max(home_section.get("pages") or []) if home_section.get("pages") else None
        if home_max_page is not None:
            for rs in related_sections:
                if rs.get("file") == home_section.get("file"):
                    rs_pages = rs.get("pages") or []
                    if rs_pages and isinstance(rs_pages[0], int) and rs_pages[0] > home_max_page:
                        future_sections.append(rs)
    except Exception:
        pass

    # Current document name (if provided)
    current_display = None
    if current_url:
        try:
            fn = current_url.split("/")[-1]
            current_display = fn.split("_", 2)[-1] if "_" in fn else fn
        except Exception:
            current_display = None

    sys_prompt = (
        "You write short, friendly, and insightful summaries grounded ONLY in the provided excerpts.\n"
        "Rules:\n"
        "- Max 120 words (aim ~100). Only exceed if absolutely necessary.\n"
        "- Use an approachable tone: fresh, helpful, and clear.\n"
        "- Cite with inline markdown links [Title](URL) to provided sections.\n"
        "- Do NOT mention page numbers; refer to titles/sections instead.\n"
        "- Assume the user has already read other PDFs, but is currently reading the open one." + (f" The current doc is '{current_display}'." if current_display else "") + "\n"
        "- If evidence is insufficient, say so briefly.\n"
    )

    def block(label: str, sec: Dict[str, Any]) -> str:
        heading_line = f"Heading: {sec.get('heading')}\n" if sec.get('heading') else ''
        pages_line = f"Pages: {sec.get('pages')}\n" if sec.get('pages') else ''
        return (
            f"[{label}] {sec.get('displayName')}\n"
            f"{heading_line}{pages_line}URL: {sec.get('fileUrl')}\n"
            "```\n" + (sec.get('text') or '') + "\n```\n"
        )

    # Replace block builder to avoid page numbers in the display
    def block_short(label: str, sec: Dict[str, Any]) -> str:
        heading_line = f"Heading: {sec.get('heading')}\n" if sec.get('heading') else ''
        link_line = f"Link: {sec.get('link')}\n" if sec.get('link') else ''
        return (
            f"[{label}] {sec.get('displayName')}\n"
            f"{heading_line}{link_line}"
            "```\n" + (sec.get('text') or '') + "\n```\n"
        )

    corpus_parts = [
        "Selected text (exact snippet):\n```\n" + sel + "\n```\n",
        "Selected paragraph (from current section):\n```\n" + (selected_para or sel) + "\n```\n",
        "Current section context:\n" + block_short("Home", home_section)
    ]
    if related_sections:
        corpus_parts.append("Related sections:\n" + "\n".join(block_short(f"Rel-{i+1}", rs) for i, rs in enumerate(related_sections)))
    if future_sections:
        corpus_parts.append("Later in this document:\n" + "\n".join(block_short(f"Later-{i+1}", rs) for i, rs in enumerate(future_sections)))

    user_prompt = "\n\n".join(corpus_parts)

    # Instruction to prefer contrasts/agreements, referencing other PDFs
    style_nudge = (
        "Write like: ‘The text you’ve selected shows … You’ve already read [Title](URL) which says … This contrasts with … Also supported by [Title](URL) … As you read on in this document, you’ll see …’. \n"
        "Keep it under the word limit unless essential."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt + "\n\n" + style_nudge}
    ]
    if stream:
        # SSE stream of LLM output if supported
        def sse_gen():
            try:
                # First, send metadata so the client can render citations early
                import json as _json
                meta_payload = {
                    "selection": sel,
                    "home": home_section,
                    "related": related_sections,
                    "provider": provider,
                }
                yield "event: meta\n"
                yield f"data: {_json.dumps(meta_payload)}\n\n"
                from llm_call import get_llm_response_stream
                for chunk in get_llm_response_stream(messages):
                    if chunk:
                        yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:  # fallback to non-stream response
                try:
                    # Also send meta first on fallback
                    import json as _json
                    meta_payload = {
                        "selection": sel,
                        "home": home_section,
                        "related": related_sections,
                        "provider": provider,
                    }
                    yield "event: meta\n"
                    yield f"data: {_json.dumps(meta_payload)}\n\n"
                    analysis = get_llm_response(messages)
                    yield f"data: {analysis}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as ee:
                    err = str(ee or e)
                    yield f"data: {{\"error\": \"{err}\"}}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(sse_gen(), media_type="text/event-stream")
    else:
        try:
            analysis = get_llm_response(messages)  # uses configured provider
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "error": f"LLM call failed: {e}",
                "selection": sel,
                "home": home_section,
                "related": related_sections,
                "provider": provider
            })

        return {
            "analysis": analysis,
            "selection": sel,
            "home": home_section,
            "related": related_sections,
            "provider": provider,
        }


@app.get("/api/library")
async def list_pdfs():
    """List uploaded PDFs in library and current folders."""
    def list_dir(dir_path: str, base_url: str):
        try:
            items = []
            for fn in os.listdir(dir_path):
                full = os.path.join(dir_path, fn)
                if not os.path.isfile(full):
                    continue
                if not fn.lower().endswith('.pdf'):
                    continue
                items.append({
                    "file": fn,
                    "url": f"{base_url}/{fn}",
                    "displayName": fn.split('_', 1)[1] if '_' in fn else fn,
                    "mtime": os.path.getmtime(full),
                })
            # sort by modified time desc
            items.sort(key=lambda x: x["mtime"], reverse=True)
            # drop mtime from output
            for it in items:
                it.pop("mtime", None)
            return items
        except Exception:
            return []

    library = list_dir(FILES_LIBRARY_DIR, "/files/library")
    current = list_dir(FILES_CURRENT_DIR, "/files/current")
    return {"library": library, "current": current}


@app.get("/api/health")
async def health():
    try:
        get_model()
        get_index()
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/api/audio")
async def generate_audio_endpoint(
    selection: str = Form(...),
    mode: str = Form("podcast"),  # 'podcast' or 'overview'
    top_k: int = Form(20),
    threshold: float = Form(0.2),
    voice_primary: Optional[str] = Form(None),
    voice_secondary: Optional[str] = Form(None),
):
    """Generate an audio overview or podcast grounded in the selection + related sections.

    Returns a JSON with audioUrl and the script used.
    """
    sel = (selection or "").strip()
    if not sel:
        return JSONResponse(status_code=400, content={"error": "selection is empty"})

    # Reuse insights retrieval to get home + related sections
    meta = load_meta()
    if not meta:
        return JSONResponse(status_code=400, content={"error": "No corpus indexed"})

    # Anchor with adaptive fallback using fused search
    _ak = _extract_acronyms_and_keywords(sel)
    anchor_queries = [sel] + _ak if _ak else [sel]
    anchor_hits = search_multi_rrf(anchor_queries, top_k=max(10, top_k), threshold=max(0.0, threshold - 0.05)) or []
    if not anchor_hits and threshold > 0.0:
        anchor_hits = search_multi_rrf(anchor_queries, top_k=max(10, top_k), threshold=max(0.0, threshold - 0.1)) or []
    if not anchor_hits and threshold > 0.0:
        anchor_hits = search_multi_rrf(anchor_queries, top_k=max(10, top_k), threshold=0.0) or []
    if not anchor_hits:
        return JSONResponse(status_code=404, content={"error": "No anchors found"})
    # Audio: apply the same overlap-aware anchor selection
    best_primary = None
    best_score = -1.0
    for cand in anchor_hits[:10]:
        sec_tmp, _ = build_section_for_hit(cand, meta)
        ov = _token_overlap(sel, sec_tmp.get("text", ""))
        name_ov = _name_overlap(sel, cand.get("file"), sec_tmp.get("displayName"), sec_tmp.get("heading"))
        ref_pen = _reference_penalty(sec_tmp.get("heading"), sec_tmp.get("text") or "")
        s = 0.8 * float(cand.get("score", 0.0)) + 0.10 * ov + 0.10 * name_ov - 0.12 * ref_pen
        if s > best_score:
            best_score = s
            best_primary = cand
    primary = best_primary or anchor_hits[0]
    home_section, _ = build_section_for_hit(primary, meta)

    query_text = _strip_citation_noise(home_section.get("text") or sel)
    heading_hint = home_section.get("heading") or ""
    q_variants = [
        query_text,
        sel,
        (heading_hint + ": " + sel) if heading_hint else sel,
    ]
    if _ak:
        q_variants.extend(_ak[:4])
    hits = search_multi_rrf(q_variants, top_k=top_k, threshold=threshold)
    if not hits and threshold > 0.0:
        hits = search_multi_rrf(q_variants, top_k=top_k, threshold=max(0.0, threshold - 0.05))
    if not hits and threshold > 0.0:
        hits = search_multi_rrf(q_variants, top_k=top_k, threshold=0.0)
    # Build distinct related sections
    def sec_key(h: Dict[str, Any]) -> Tuple[str, str]:
        file_key = str(h.get("file") or "")
        sec = h.get("heading") or f"PAGE-{h.get('page')}"
        return (file_key, str(sec))

    home_key = sec_key(primary)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for h in hits[1:]:
        k = sec_key(h)
        if k == home_key:
            continue
        grouped.setdefault(k, []).append(h)
    # Section-level ranking (same as insights)
    def section_score(section_hits: List[Dict[str, Any]]) -> float:
        scores = sorted([h["score"] for h in section_hits], reverse=True)
        mx = scores[0] if scores else 0.0
        avg_top2 = sum(scores[:2]) / min(2, len(scores)) if scores else 0.0
        return 0.7 * mx + 0.3 * avg_top2

    tmp_ranked = []
    for key, hs in grouped.items():
        s = section_score(hs)
        if key[0] == home_section.get("file"):
            s += 0.02
        tmp_ranked.append((key, hs, s))
    ranked_sections = sorted(tmp_ranked, key=lambda x: x[2], reverse=True)

    related_sections: List[Dict[str, Any]] = []
    count = 0
    seen_keys: set = set()
    seen_text_fingerprints: set = set()
    for (file, heading_or_page), _hits, _score in ranked_sections:
        if count >= 3:
            break
        if (file, heading_or_page) in seen_keys:
            continue
        tmp_primary: Dict[str, Any] = {"file": file, "heading": None, "page": None}
        if heading_or_page.startswith("PAGE-"):
            try:
                tmp_primary["page"] = int(heading_or_page.split("-", 1)[1])
            except Exception:
                tmp_primary["page"] = None
        else:
            tmp_primary["heading"] = heading_or_page
        sec, _ = build_section_for_hit(tmp_primary, meta)
        if sec.get("text"):
            import hashlib
            fp = hashlib.md5((sec.get("text") or "").strip().encode("utf-8")).hexdigest()
            if fp in seen_text_fingerprints:
                continue
            # cap text to keep TTS manageable
            sec["text"] = (sec["text"] or "")[:1500]
            related_sections.append(sec)
            seen_keys.add((file, heading_or_page))
            seen_text_fingerprints.add(fp)
            count += 1

    # Build a podcast/overview script via LLM
    provider = os.getenv("LLM_PROVIDER", "gemini")
    def fmt_section(sec: Dict[str, Any]) -> str:
        title = sec.get("displayName") or sec.get("file")
        head = sec.get("heading")
        return f"{title} - {head}\n" if head else f"{title}\n"

    # Generate a compact insights summary to seed the audio (3-5 bullets)
    try:
        insights_messages = [
            {"role": "system", "content": "Extract 3-5 concise, evidence-grounded bullets: agreements, contrasts, extensions, or caveats. Use only the provided texts."},
            {"role": "user", "content": (
                "Selected text:\n" + sel + "\n\n" +
                "Current section:\n" + (home_section.get("text") or "") + "\n\n" +
                ("Related sections:\n" + "\n\n".join((rs.get("text") or "") for rs in related_sections) if related_sections else "")
            )}
        ]
        insights_summary = get_llm_response(insights_messages)
    except Exception:
        insights_summary = None

    script_prompt = (
        "Create a concise audio script "
        + ("for a 2-5 minute podcast between two speakers" if mode == "podcast" else "as a 2-3 minute narrated overview")
        + ". Ground it ONLY in these excerpts. Include contrasts, extensions, or caveats when present. Keep language clear and engaging.\n\n"
        "Selected text:\n" + sel + "\n\n"
        "Current section:\n" + (home_section.get("text") or "") + "\n\n"
        + ("Related sections:\n" + "\n\n".join((rs.get("text") or "") for rs in related_sections) if related_sections else "") + "\n\n"
        + ("Insights summary (bullets):\n" + insights_summary if insights_summary else "")
    )

    if mode == "podcast":
        script_style = (
            "Write dialogue as SPEAKER 1 and SPEAKER 2, alternating short turns (1-3 sentences each). "
            "Open with a one-sentence hook, then explain the main idea, compare with related work, and end with a crisp takeaway."
        )
    else:
        script_style = (
            "Write as a single narrator. Use a brief intro, 2-3 short paragraphs of explanation and contrast, then a one-sentence takeaway."
        )

    messages = [
        {"role": "system", "content": "You create short, engaging, and accurate audio scripts grounded only in provided text."},
        {"role": "user", "content": script_prompt + "\n\n" + script_style}
    ]

    script_text = get_llm_response(messages)
    if not script_text or not isinstance(script_text, str):
        return JSONResponse(status_code=500, content={"error": "Failed to generate script"})

    # Split into segments if podcast format
    audio_dir = os.path.join(DATA_DIR, "files", "audio")
    file_name = f"audio_{uuid.uuid4().hex}.mp3"
    out_path = os.path.join(audio_dir, file_name)

    if mode == "podcast":
        # Parse lines starting with SPEAKER 1:/SPEAKER 2: (or similar)
        import re
        segments: List[Dict[str, str]] = []
        for line in script_text.splitlines():
            m = re.match(r"\s*(SPEAKER\s*1|SPEAKER\s*2|HOST|GUEST)\s*[:\-]\s*(.*)", line, re.IGNORECASE)
            if m:
                who = m.group(1).upper().replace(" ", "")
                txt = m.group(2).strip()
                if txt:
                    segments.append({"speaker": who, "text": txt})
        if not segments:
            # fallback: alternate by sentences
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script_text) if s.strip()]
            for i, s in enumerate(sents):
                who = "SPEAKER1" if i % 2 == 0 else "SPEAKER2"
                segments.append({"speaker": who, "text": s})

        voice_map = {}
        if voice_primary:
            voice_map["SPEAKER1"] = voice_primary
            voice_map["SPEAKER 1"] = voice_primary
        if voice_secondary:
            voice_map["SPEAKER2"] = voice_secondary
            voice_map["SPEAKER 2"] = voice_secondary

        try:
            generate_audio_podcast(segments, out_path, provider=os.getenv("TTS_PROVIDER", "local"), voice_map=voice_map)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Audio synthesis failed: {e}"})
    else:
        try:
            generate_audio(script_text, out_path, provider=os.getenv("TTS_PROVIDER", "local"), voice=voice_primary)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Audio synthesis failed: {e}"})

    return {"audioUrl": f"/files/audio/{file_name}", "script": script_text, "mode": mode}
