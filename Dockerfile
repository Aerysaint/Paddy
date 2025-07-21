# Use Python 3.9-slim as base image with AMD64 platform specification
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal system dependencies required for PyMuPDF (CPU-only)
# Keep minimal to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with CPU-only optimizations
# Use PyTorch CPU-only index to avoid CUDA dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py .

# Create input and output directories for volume mounts
RUN mkdir -p /app/input /app/output

# Set proper permissions
RUN chmod -R 755 /app

# Configure for offline operation - no network access needed after build
# All dependencies are installed during build phase

# Set up proper entry point
ENTRYPOINT ["python", "main.py"]

# Default command arguments for container operation
CMD ["-i", "/app/input", "-o", "/app/output"]

# Health check to verify container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); from src.pdf_extractor import PDFExtractor; print('OK')" || exit 1

# Labels for container metadata
LABEL maintainer="PDF Outline Extractor"
LABEL description="Extract document structure from PDF files"
LABEL version="1.0"

# Volume mount points for input and output
VOLUME ["/app/input", "/app/output"]

# Expose no ports (this is a batch processing application)