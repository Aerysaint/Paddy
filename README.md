# Paddy — PDF Selection-to-Search RAG (Retrieval-Augmented Guidance)

This repository implements a fast, evidence-grounded PDF reader that locates a user's selected text in an indexed corpus, expands to the surrounding section, finds related sections across the corpus, and (optionally) synthesizes a short, citation-aware summary or audio script.

The core idea: prefer fast, embedding-based retrieval and smart heuristics to anchor selections and produce grounded outputs quickly, using LLMs sparingly only for concise analysis or TTS script generation. This keeps the system responsive while maintaining reasonable accuracy.

## Quick start — Docker

The project aims to run in Docker. Example commands tested by the author:

```powershell
docker build -t my_app .

docker run -d -p 8080:8080 -p 8000:8000 --name paddy-container1 \
	-e ADOBE_EMBED_API_KEY=b8c27736e8364a92bb0ebbc65553d934 \
	-e GOOGLE_API_KEY= <your api key> \
	-e GEMINI_MODEL=gemini-2.5-flash \
	-e LLM_PROVIDER=gemini \
	-e TTS_PROVIDER=azure \
	-e AZURE_TTS_KEY=dummy \
	-e AZURE_TTS_ENDPOINT=alsodummy \
	my_app
```

Ports exposed in the above example:
- Backend API (FastAPI / Uvicorn): port `8000`
- Frontend (Next.js or a built static host inside the container): `8080` (mapped in the example run)

I've tried my best for it to run in Docker, but in case it throws an error, please run the services locally as a fallback (recommended for development and debugging):

```powershell
# Backend (in backend/)
cd backend
# create virtualenv if desired, then
py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Frontend (from repo root or frontend/)
cd frontend
npm install
npm run dev
```

## What this repo contains (high level)
- `backend/` — FastAPI backend. Key modules:
	- `app.py` — main API: ingesting PDFs, indexing, search/insights, audio endpoints.
	- `llm_call.py` — thin multi-provider LLM wrapper (Google Gemini, Azure, OpenAI, Ollama).
	- `generate_audio.py` — multi-provider TTS orchestration (Azure, Google Cloud, local `espeak-ng`).
- `frontend/` — Next.js frontend (viewer, UI components).
- `Dockerfile` — container build instructions.


## API endpoints (high-level)
- `POST /api/ingest` — upload PDF(s), parse/ocr, chunk, embed, and add to FAISS index.
- `POST /api/current` — set the currently open PDF for the viewer.
- `POST /api/search` — simple similarity search over indexed chunks.
- `POST /api/insights` — anchor a textual selection and return a compact, grounded analysis (calls LLM).
- `POST /api/audio` — build an audio script (LLM) and synthesize it with configured TTS provider.
- `GET /api/library` — list available PDFs (library + current).
- `GET /api/health` — health check that attempts to initialize model/index.

## Methodology & technical pitch (detailed)

This section explains the design decisions and tradeoffs in depth. If you want a short summary: we built a retrieval-first pipeline using fast embeddings (hashing by default) + FAISS plus light heuristics (headings, RRF fusion, de-duplication). LLMs are used sparingly to summarize, contrast, and generate audio scripts. This choice optimizes responsiveness and cost while keeping outputs grounded in actual document text.

1) Retrieval-first (why): speed, cost, and provenance

	- Latency: calling modern LLMs (especially high-quality multi-billion-parameter models) for every small user interaction produces slow responses (seconds -> many seconds). That is incompatible with a fluid reader experience.
	- Cost: repeated LLM calls are expensive. Using embeddings + a local vector index reduces the number of LLM calls to just the moments where a small concise synthesis is required.
	- Provenance: users need grounded answers that cite passages. Retrieval-first means the system returns actual document excerpts and constructs the answer from them — the LLM's role is to paraphrase and package evidence, not to invent.

2) How we get both speed and reasonable accuracy

	- HashingVectorizer for embeddings-by-default: `USE_HASH_EMBED=1`
		- Fast, memory-compact, and CPU-friendly. Produces high-dimensional sparse vectors which we l2-normalize for cosine similarity in FAISS.
		- This yields very fast indexing and query times even on modest machines without GPUs.
		- Accuracy is lower than SBERT, but acceptable for a first-pass retrieval layer and much faster.

	- Optional SBERT (sentence-transformers) path:
		- If you need higher semantic accuracy and have the resources, set `USE_HASH_EMBED=0` and install `requirements-sbert.txt` to enable SBERT embeddings (model controlled by `EMBED_MODEL`).

	- FAISS IndexFlatIP on normalized vectors:
		- Inner-product search on normalized vectors == cosine similarity. FAISS keeps queries sub-second even for tens of thousands of chunks.

	- Chunking + heading heuristics:
		- PDFs are parsed per page; text is chunked into overlapping windows (default chunk ~=1000 chars, overlap 200) so that local context is preserved.
		- Heuristics detect headings (Title Case or ALL CAPS short lines) and propagate heading context to chunks; this improves ability to anchor a selection to the correct section.

	- Adaptive multi-query + Reciprocal Rank Fusion (RRF):
		- For a user selection, the system builds query variants (selection, selected paragraph, heading hints, extracted acronyms/keywords like "ViT", "GAN") and runs multiple searches.
		- Those result lists are fused with RRF to boost documents that appear consistently across variants. This improves recall and robustness without heavy LLM involvement.

	- Section assembly and re-ranking:
		- The anchor chunk found is expanded into a 'section' by joining all chunks from the same file+heading (or file+page). This produces a coherent textual unit for downstream use.
		- Candidate related sections are de-duplicated (md5 fingerprints), penalized heuristically if they look like references/bibliographies, and ranked combining embedding score and lexical overlap with the selection.

	- Conservative LLM role:
		- LLMs are used only to produce short, user-facing artifacts:
			* 100–120 word analyses (insights)
			* 2–5 minute audio scripts (podcast or overview)
		- The prompts explicitly force the model to be evidence-grounded and to cite only the provided excerpts. The server returns the extracted excerpts and links alongside the LLM output so UI can display provenance.

3) Tradeoffs and practical considerations

	- Speed vs semantic precision: HashingVectorizer + FAISS is orders of magnitude faster to start and query than SBERT with GPU or even CPU SBERT for some scales. We accept a modest drop in fine-grained semantic matching because RRF and query expansions recover much of the needed recall for the reader scenario.

	- When to flip to heavy LLM usage: for deeper analysis tasks (long multi-step reasoning, code synthesis, or heavy summarization across many documents), consider using SBERT embeddings + a single LLM call, or even a hybrid approach where top-N documents are re-scored with a more expensive model.

	- Budgeting LLM calls: the architecture intentionally limits LLM usage to a single call per insights/audio request. This keeps costs predictable. If streaming is enabled, the client receives a quick metadata event (home/related sections) immediately for fast UI rendering.

4) Robustness heuristics (practical engineering wins)

	- OCR fallback: if a PDF page yields very little text via PyMuPDF, we render the page to an image and run Tesseract OCR. This allows scanned PDFs to still be searchable.
	- Citation/noise stripping: candidate section text is sanitized to remove bracket citations (`[12]`), DOIs, arXiv IDs, and URLs before feeding queries to the index or LLM.
	- Reference penalty: sections that look like bibliographies are demoted automatically to avoid returning references as top hits.



5) When the retrieval-first approach might fail

	- Extremely subtle paraphrase retrieval: if the user's selection is phrased very differently from text in the corpus, hashing embeddings can miss it. SBERT helps here.
	- Very small corpora or extreme sparsity: if only a few tokens match and headings are missing, heuristics may fail to anchor correctly.
	- Missing or poor-quality OCR on scanned documents: some papers with poor scans will need manual intervention.

Recommended mitigations: enable SBERT when higher precision is required, tune `HASH_DIM`, increase `top_k`, or slightly lower similarity thresholds in `app.py`.

## Troubleshooting & dev tips

- If Docker build fails: run the backend and frontend locally (see commands above). Errors during Docker runs often come from native dependencies (Faiss, PyMuPDF) or system-level tools (ffmpeg, tesseract). Fix those on the host and re-build.
- Use `GET /api/health` to see whether the model/index initialized correctly.
- If you want better retrieval quality quickly: set `USE_HASH_EMBED=0` and install the SBERT extras (`backend/requirements-sbert.txt`) then restart.

## Notes on reproducibility and testing
- The repository includes a `backend/tests/` harness for offline evaluation and judge runs (useful when tuning retrieval/thresholds). Look in `backend/tests` for example usage patterns.

## Final words
This project targets a practical middle-ground: keep interactions fast and inexpensive by making the retrieval layer do the heavy lifting, then use modern LLMs judiciously to produce concise human-friendly outputs that are explicitly tied to document excerpts. That combination provides a snappy, grounded reading experience without excessive latency or cost.

If anything in the Docker setup fails on your machine, please run the backend with `uvicorn app:app` from the `backend` folder and run the frontend with `npm run dev` from the `frontend` folder.

Thanks a lot, it was a pleasure working on this project.
