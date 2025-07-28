# Challenge 1A: PDF Heading Extraction

This solution extracts structured headings and titles from PDF documents using advanced text analysis and formatting detection techniques.

## Overview

The system processes PDF files to identify document titles and hierarchical heading structures (H1, H2, H3, etc.) based on:
- Font size analysis and frequency patterns
- Text formatting (bold, italic, underlined, all caps)
- Numbering patterns (1, 1.1, 1.1.1, etc.)
- Spatial layout and positioning
- Content validation and filtering

## Features

- **Intelligent Title Detection**: Identifies document titles using multi-criteria scoring
- **Hierarchical Heading Extraction**: Detects H1, H2, H3+ levels based on numbering and formatting
- **Robust Text Validation**: Filters out table content, mathematical expressions, and invalid text
- **Unicode Support**: Handles international characters and mathematical symbols
- **JSON Output**: Structured output format for easy integration

## Architecture

### Core Components

1. **`extractor.py`**: PDF text extraction with detailed formatting information
2. **`get_stats.py`**: Font size analysis and statistical processing
3. **`heading_extractor.py`**: Main heading detection logic using human-like analysis
4. **`run.py`**: Docker container entry point for batch processing

### Processing Pipeline

```
PDF Input → Text Extraction → Font Analysis → Title Detection → Heading Classification → JSON Output
```

## Usage

### Docker (Recommended)

1. **Build the container:**
   ```bash
   docker build -t pdf-heading-extractor .
   ```

2. **Run with input/output directories:**
   ```bash
   docker run -v /path/to/input:/app/input -v /path/to/output:/app/output pdf-heading-extractor
   ```

### Direct Python Usage

```bash
python heading_extractor.py document.pdf -o output.json
```

## Input/Output Format

### Input
- PDF files in the mounted `/app/input` directory

### Output
- JSON files with the same base name as input PDFs
- Structure:
  ```json
  {
    "title": "Document Title",
    "outline": [
      {
        "level": "H1",
        "text": "1 Introduction",
        "page": 2
      },
      {
        "level": "H2", 
        "text": "1.1 Background",
        "page": 2
      }
    ]
  }
  ```

## Algorithm Details

### Title Detection Strategy

1. **Font Size Analysis**: Identifies rare, large font sizes as title candidates
2. **Multi-Criteria Scoring**: Evaluates based on:
   - Font size relative to body text
   - Formatting (bold, centered, all caps)
   - Document position (early appearance)
   - Text length (reasonable title length)

### Heading Classification

1. **Numbering Pattern Detection**: Recognizes hierarchical numbering (1, 1.1, 1.1.1)
2. **Font-Based Hierarchy**: Groups similar font sizes and assigns levels
3. **Content Validation**: Filters out:
   - Table of contents entries
   - Mathematical expressions
   - Pure numeric content
   - Bullet points and list items

### Text Validation Logic

The system uses sophisticated validation to distinguish between:
- **Valid headings**: "1 Introduction", "Background", "3.2 Methodology"
- **Invalid content**: "1 + 2 = 3", "• Item 1", "Table 1.1"

## Dependencies

- **PyMuPDF (fitz)**: PDF text extraction and analysis
- **Python 3.9+**: Core runtime environment
- **Standard libraries**: re, json, collections, statistics

## Performance Characteristics

- **Processing Speed**: ~1-5 seconds per typical research paper
- **Memory Usage**: Low memory footprint, processes documents incrementally
- **Accuracy**: High precision on academic papers and technical documents
- **Scalability**: Batch processing support via Docker container

## Error Handling

- Graceful handling of corrupted PDFs
- Fallback mechanisms for edge cases
- Detailed logging and error reporting
- Unicode encoding safety measures

## Limitations

- Optimized for academic/technical documents with clear heading structures
- May struggle with heavily formatted documents or unusual layouts
- Requires clear font size differentiation for optimal results
- Table detection relies on PyMuPDF's built-in capabilities

## Testing

The system has been tested on various document types including:
- Academic research papers (e.g., "Attention Is All You Need")
- Technical reports and specifications
- Multi-language documents with Unicode content
- Documents with mathematical notation

## Configuration

Key parameters can be adjusted in the code:
- Font size tolerance for grouping similar sizes
- Minimum heading score thresholds
- Text length limits for validation
- Spacing analysis parameters

## Troubleshooting

### Common Issues

1. **No headings detected**: Check if document has clear font size hierarchy
2. **Wrong heading levels**: Verify numbering patterns are consistent
3. **Missing content**: Ensure PDF has extractable text (not scanned images)
4. **Unicode errors**: System handles most Unicode characters automatically

### Debug Mode

Run with verbose output to see detailed analysis:
```bash
python heading_extractor.py document.pdf -o output.json --verbose
```

## Future Enhancements

- Machine learning-based heading detection
- Support for scanned documents (OCR integration)
- Advanced table structure recognition
- Multi-column layout handling
- Custom formatting rule configuration

## License

This solution is designed for the Adobe's Conect the Dots challenge and demonstrates advanced text analysis techniques for document structure extraction.