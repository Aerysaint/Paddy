# Start with a base Python image compatible with linux/amd64
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install NumPy with version constraint first to avoid compatibility issues
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# Install PyTorch CPU-only versions
RUN pip install --no-cache-dir torch==2.1.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install all other Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container's working directory
# This includes main.py, chunker.py, reranker.py, etc.
COPY . .

# This is the command that will be executed when the container starts
CMD ["python", "run.py"]