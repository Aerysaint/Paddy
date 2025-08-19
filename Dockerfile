# Use Python base image
FROM python:3.11-slim

# Install Node.js, npm, and Tesseract OCR
RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend requirements first for better caching
COPY backend/requirements.txt ./backend/
COPY frontend/package.json frontend/package-lock.json* ./frontend/

# Install dependencies
WORKDIR /app/backend
RUN pip install -r requirements.txt

WORKDIR /app/frontend
RUN npm install

# Copy the rest of the application
WORKDIR /app
COPY . .

# Set environment variables
ENV FRONTEND_ORIGIN=http://localhost:3000


# Install Python dependencies
WORKDIR /app/backend
RUN pip install -r requirements.txt

# Install Node.js dependencies
WORKDIR /app/frontend
RUN npm install

# Create startup script
WORKDIR /app
RUN echo '#!/bin/bash\n\
# Start backend server in background\n\
cd /app/backend\n\
uvicorn app:app --reload --host 0.0.0.0 --port 8000 &\n\
\n\
# Start frontend server in background\n\
cd /app/frontend\n\
npm run dev -- --hostname 0.0.0.0 --port 3000 &\n\
\n\
# Wait for any process to exit\n\
wait\n' > start.sh && chmod +x start.sh

# Expose ports
EXPOSE 8000 3000

# Start both services
CMD ["./start.sh"]