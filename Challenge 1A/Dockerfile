# Start with a base Python image. This supports the required amd64 architecture[cite: 56, 57].
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python libraries your script needs [cite: 86]
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (your .py scripts) into the container
COPY . .

# This is the command that will be executed when the container starts
CMD ["python", "run.py"]