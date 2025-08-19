## Containerize backend (FastAPI on 8000) and frontend (Next.js on 8080)
## No source changes; everything wired in Docker only.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
		PYTHONUNBUFFERED=1 \
		PIP_NO_CACHE_DIR=1 \
		DEBIAN_FRONTEND=noninteractive

# OS deps: Node.js 20.x, ffmpeg (for pydub), tesseract-ocr (OCR), espeak-ng (local TTS), build tools
RUN set -eux; \
		apt-get update; \
		apt-get install -y --no-install-recommends curl gnupg ca-certificates; \
		curl -fsSL https://deb.nodesource.com/setup_20.x | bash -; \
		apt-get install -y --no-install-recommends \
			nodejs \
			ffmpeg \
			tesseract-ocr \
		bash \
			espeak-ng \
			git \
			build-essential; \
		rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend dependencies first for layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Frontend dependencies
COPY frontend/package.json /app/frontend/package.json
RUN set -eux; cd /app/frontend; npm install --no-audit --no-fund

# Copy application code
COPY backend /app/backend
COPY frontend /app/frontend

# Entrypoint script to start both services
RUN cat > /entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Map ADOBE_EMBED_API_KEY to the public Next env expected by the app
export NEXT_PUBLIC_ADOBE_CLIENT_ID="${NEXT_PUBLIC_ADOBE_CLIENT_ID:-${ADOBE_EMBED_API_KEY:-}}"
# Origins used by backend for link generation
export FRONTEND_ORIGIN="${FRONTEND_ORIGIN:-http://localhost:8080}"
export BACKEND_ORIGIN="${BACKEND_ORIGIN:-http://localhost:8000}"

echo "[entrypoint] Starting FastAPI backend on 0.0.0.0:8000"
cd /app/backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 &

echo "[entrypoint] Starting Next.js frontend on 0.0.0.0:8080 (dev mode)"
cd /app/frontend
# Dev mode reads NEXT_PUBLIC_* from runtime env without baking into the image
npx next dev -H 0.0.0.0 -p 8080
EOF

RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8080 8000

CMD ["/entrypoint.sh"]

# Build: docker build --platform linux/amd64 -t yourimageidentifier .
# Run:   docker run -v /path/to/credentials:/credentials \
#              -e ADOBE_EMBED_API_KEY=... -e LLM_PROVIDER=gemini \
#              -e GOOGLE_APPLICATION_CREDENTIALS=/credentials/adbe-gcp.json \
#              -e GEMINI_MODEL=gemini-2.5-flash -e TTS_PROVIDER=azure \
#              -e AZURE_TTS_KEY=TTS_KEY -e AZURE_TTS_ENDPOINT=TTS_ENDPOINT \
#              -p 8080:8080 yourimageidentifier
# NOTE: The frontend code calls http://localhost:8000 from the browser. To use the
# backend inside this container, you will also need to publish port 8000:
#              -p 8000:8000

