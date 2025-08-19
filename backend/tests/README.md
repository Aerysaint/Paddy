# Retrieval Test Harness

A tiny, CPU-only harness to evaluate relevant-sections retrieval end-to-end using real PDFs from arXiv.

What it does:
- Searches arXiv for a topic
- Downloads a few PDFs
- Ingests them into the running backend
- Runs the Insights endpoint on a few heuristic selections and prints a concise summary

## Prereqs
- Backend server running locally (defaults to http://localhost:8000)
- Python packages: `requests`, `feedparser`

On Windows PowerShell:
```powershell
# (optional) use repo venv
python -m venv .venv ; .\.venv\Scripts\Activate.ps1

# install deps for harness
pip install requests feedparser

# ensure backend deps are installed and server is running
# pip install fastapi uvicorn[standard] numpy faiss-cpu scikit-learn PyMuPDF pillow pytesseract python-dotenv
# uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

## Run
```powershell
python backend/tests/retrieval_harness.py --topic "large language models" --n 3
```

Optional flags:
- `--stream` to stream the insights output
- `--threshold` to adjust similarity threshold (default 0.2)
- `--top_k` to tweak candidate depth (default 20)

Outputs:
- Ingest response count
- Home/Related section titles and headings
- Truncated analysis text

> Note: arXiv’s API is public but rate-limited. Keep `--n` small for quick runs.
