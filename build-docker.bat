@echo off
REM Build script for PDF Outline Extractor Docker container
REM This script builds the Docker container with proper platform specification

echo Building PDF Outline Extractor Docker container...
echo Platform: linux/amd64 (CPU-only)
echo Base image: python:3.9-slim
echo.

REM Build the container
docker build --platform linux/amd64 -t pdf-outline-extractor .

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo Build completed successfully!
echo.

REM Show container information
echo Container information:
docker images pdf-outline-extractor

echo.
echo Testing container...

REM Test help command
echo Testing help command:
docker run --rm pdf-outline-extractor --help

echo.
echo Testing PyTorch CPU-only installation:
docker run --rm --entrypoint python pdf-outline-extractor -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"

echo.
echo Container build and test completed successfully!
echo.
echo Usage examples:
echo   # Process PDFs from ./pdfs directory to ./outputs directory:
echo   docker run --rm -v "%CD%/pdfs:/app/input" -v "%CD%/outputs:/app/output" pdf-outline-extractor
echo.
echo   # Process single file:
echo   docker run --rm -v "%CD%/pdfs:/app/input" -v "%CD%/outputs:/app/output" pdf-outline-extractor --single-file /app/input/file.pdf
echo.
echo   # Verbose logging:
echo   docker run --rm -v "%CD%/pdfs:/app/input" -v "%CD%/outputs:/app/output" pdf-outline-extractor -v