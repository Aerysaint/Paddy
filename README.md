# Experiment Adobe – PDF Selection-to-Search RAG

End-to-end prototype with:
- Python FastAPI backend for bulk PDF ingestion (with PyMuPDF parsing + optional Tesseract OCR), chunking, embeddings (CPU-only HashingVectorizer by default, optional SentenceTransformers), and FAISS vector DB.
- Next.js frontend with Adobe PDF Embed API to render the current PDF in high fidelity and a slick, scrollable bubble showing vector search hits for selected text.

## Quick start

Prereqs:
- Python 3.10+
- Node.js 18+
- (Optional OCR) Tesseract installed and on PATH (Windows installer: https://github.com/UB-Mannheim/tesseract/wiki)

Environment variables:
- Frontend: `NEXT_PUBLIC_ADOBE_CLIENT_ID` (Adobe PDF Embed API client ID)
- Backend: optional `TESSERACT_PATH` to point to tesseract.exe

### Backend (FastAPI, CPU-only by default)

1) Create venv and install deps

```powershell
cd c:\Users\tejas\experiment_adobe\backend
py -3 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # CPU-only
```

2) (Optional) Enable higher-quality embeddings with SentenceTransformers (still CPU-only)

```powershell
pip install -r requirements-sbert.txt
$env:USE_HASH_EMBED="0"  # switch to SBERT
```

3) Run API

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- Static files served at: http://localhost:8000/files/
- FAISS index is persisted under `backend/data/store/`

### Frontend (Next.js)

1) Install deps and run

```powershell
cd c:\Users\tejas\experiment_adobe\frontend
npm install
$env:NEXT_PUBLIC_ADOBE_CLIENT_ID="YOUR_ADOBE_CLIENT_ID"
npm run dev
```

Open http://localhost:3000

### Workflow
- Upload previously-read PDFs in the Bulk Ingest card.
- Upload the Current PDF (the one you’re reading). The viewer loads it via Adobe Embed.
- Select text in the viewer. A floating bubble appears near your selection with the top-k matches from your library.

## Notes
- Offline: All NLP and vector ops run locally. By default, embeddings use CPU-only hashing (no model downloads). To use SentenceTransformers, install `requirements-sbert.txt` and set `USE_HASH_EMBED=0` (first run will download and cache the model in `%USERPROFILE%/.cache/torch/sentence_transformers`).
- OCR: If a page has little/no extractable text, we rasterize and OCR the page. Install Tesseract for better coverage of scanned PDFs.
- Headings: We record a simple heuristic heading per chunk (nearest preceding Title-Case line or PDF outline entry if available).

## Next steps
- Add GPU-accelerated embeddings if available.
- Add better structural parsing (font-size aware headings via PyMuPDF spans).
- Add tests and CI.
