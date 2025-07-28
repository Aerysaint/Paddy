# Challenge 1B: Persona-Based Document Retrieval and Reranking

A sophisticated document retrieval system that combines bi-encoder semantic search with cross-encoder reranking to deliver highly relevant, contextually appropriate results for specific user personas and job-to-be-done scenarios.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker installed on your system
- Input JSON file in the expected format
- PDF documents to process

### Running the Container

1. **Build the Docker image:**
   ```bash
   docker build -t challenge1b .
   ```

2. **Prepare your input:**
   - Create an input directory with your challenge input JSON file
   - Ensure your PDFs are in a `PDFs` subdirectory relative to the input JSON
   
   Example structure:
   ```
   input/
   ├── challenge1b_input.json
   └── PDFs/
       ├── document1.pdf
       ├── document2.pdf
       └── ...
   ```

3. **Run the container:**
   ```bash
   docker run -v /path/to/your/input:/app/input -v /path/to/your/output:/app/output challenge1b
   ```

   Replace `/path/to/your/input` and `/path/to/your/output` with your actual paths.

### Example Usage

```bash
# Build the image
docker build -t challenge1b .

# Run with mounted volumes
docker run \
  -v $(pwd)/Collection\ 1:/app/input \
  -v $(pwd)/output:/app/output \
  challenge1b
```

This will:
- Process the JSON input file found in the input directory
- Extract and process all PDFs listed in the input
- Generate comprehensive, persona-based search results
- Save the output as `challenge1b_output.json` in the output directory

## 📋 Input Format

The system expects a JSON input file with the following structure:

```json
{
    "challenge_info": {
        "challenge_id": "round_1b_002",
        "test_case_name": "travel_planner",
        "description": "France Travel"
    },
    "documents": [
        {
            "filename": "Document1.pdf",
            "title": "Document 1 Title"
        },
        {
            "filename": "Document2.pdf", 
            "title": "Document 2 Title"
        }
    ],
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a trip of 4 days for a group of 10 college friends."
    }
}
```

## 📤 Output Format

The system generates a comprehensive JSON output with:

```json
{
    "metadata": {
        "input_documents": ["Document1.pdf", "Document2.pdf"],
        "persona": "Travel Planner",
        "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
        "processing_timestamp": "2025-07-28T12:00:00.000000"
    },
    "extracted_sections": [
        {
            "document": "Document1.pdf",
            "section_title": "Travel Planning Guide",
            "importance_rank": 1,
            "page_number": 1
        }
    ],
    "subsection_analysis": [
        {
            "document": "Document1.pdf",
            "refined_text": "Comprehensive content with detailed information...",
            "page_number": 1
        }
    ]
}
```

## 🏗️ System Architecture

### Core Components

1. **Document Processing Pipeline**
   - PDF text extraction with Unicode support
   - Intelligent heading detection
   - Content chunking and aggregation

2. **Retrieval System**
   - Bi-encoder for fast initial retrieval (BAAI/bge-small-en-v1.5)
   - Cross-encoder for precise reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)

3. **Persona Integration**
   - Role-based context understanding
   - Job-to-be-done task alignment
   - Contextual query enhancement

### Processing Flow

```
Input JSON → PDF Processing → Content Chunking → Index Building → 
Persona Query → Bi-encoder Retrieval → Cross-encoder Reranking → 
Formatted Output
```

## 🔧 Configuration Options

The system can be configured through command-line arguments in `run.py`:

- `--final-results`: Number of final results to return (default: 5)
- `--initial-candidates`: Number of initial candidates for reranking (default: 50)
- `--retriever-model`: Bi-encoder model for initial retrieval
- `--reranker-model`: Cross-encoder model for reranking

## 📊 Performance Characteristics

- **Processing Speed**: ~3 seconds per document
- **Memory Usage**: Optimized for large document collections
- **Accuracy**: High-precision results through two-stage retrieval
- **Scalability**: Handles hundreds of documents efficiently
- **CPU Optimization**: Uses PyTorch CPU-only builds for smaller container size and broader compatibility

## 🛠️ Development Setup

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run directly:**
   ```bash
   python main.py input.json --final-results 5 --output-file output.json
   ```

### Key Files

- `main.py`: Main processing pipeline with challenge mode support
- `heading_extractor.py`: PDF heading detection and structure analysis
- `chunker.py`: Content chunking with intelligent aggregation
- `retriever.py`: Bi-encoder semantic search implementation
- `reranker.py`: Cross-encoder reranking for improved relevance
- `run.py`: Docker container entry point

## 🔍 Features

### Advanced Document Processing
- **Unicode Support**: Handles international text and special characters
- **Intelligent Chunking**: Creates meaningful content sections
- **Structure Preservation**: Maintains document hierarchy and context
- **Error Recovery**: Graceful handling of problematic documents

### Sophisticated Retrieval
- **Hybrid Architecture**: Combines speed of bi-encoders with accuracy of cross-encoders
- **Persona Awareness**: Tailors results to specific user roles and tasks
- **Contextual Understanding**: Deep semantic analysis of query-document relationships
- **Importance Ranking**: Results ordered by true relevance, not just similarity

### Production Ready
- **Docker Support**: Containerized deployment for consistent environments
- **Comprehensive Logging**: Detailed processing logs for monitoring
- **Robust Error Handling**: Continues processing even when individual documents fail
- **Performance Optimization**: Efficient processing of large document collections

## 📚 Technical Details

### Models Used
- **Retriever**: BAAI/bge-small-en-v1.5 (384-dimensional embeddings)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Platform**: CPU-optimized for broad compatibility (uses PyTorch CPU-only builds)

### Dependencies
- PyMuPDF: PDF processing and text extraction
- sentence-transformers: Embedding models and cross-encoders
- torch: Deep learning framework (CPU-only version)
- transformers: Model loading and inference
- numpy: Pinned to 1.x for compatibility with PyTorch modules

## 🐛 Troubleshooting

### Common Issues

1. **NumPy Compatibility**: System uses NumPy 1.x for compatibility with PyTorch modules
2. **Unicode Errors**: The system includes robust Unicode handling for international text
3. **Memory Issues**: Adjust batch sizes in the configuration if needed
4. **PDF Processing Failures**: Check PDF format and accessibility
5. **Model Download**: First run may take time to download models

### Logs and Debugging

The system provides comprehensive logging:
- Document processing progress
- Chunk creation statistics
- Retrieval and reranking metrics
- Error details and recovery actions

## 📈 Performance Optimization

### For Large Document Collections
- Use SSD storage for faster I/O
- Increase available RAM for better caching
- Consider GPU acceleration for faster inference
- Batch process multiple documents simultaneously

### Memory Management
- The system is optimized for CPU processing
- Memory usage scales with document collection size
- Automatic cleanup of intermediate results

## 🤝 Contributing

For development and contributions:
1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure Docker compatibility

## 📄 License

This project is part of Challenge 1B submission and follows the challenge guidelines and requirements.

---

For detailed technical information about our approach, see [approach_explanation.md](approach_explanation.md).