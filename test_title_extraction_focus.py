"""
Focused test for title extraction to understand the discrepancies and fix them.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.pdf_extractor import PDFExtractor
from src.title_extractor import TitleExtractor
from src.logging_config import setup_logging

# Set up logging
logger = setup_logging('INFO')


def analyze_title_extraction():
    """Analyze what our title extractor produces vs expected titles."""
    
    pdfs_dir = Path("pdfs")
    expected_outputs_dir = Path("outputs")
    
    if not pdfs_dir.exists() or not expected_outputs_dir.exists():
        logger.error("Required directories not found!")
        return
    
    # Initialize components
    pdf_extractor = PDFExtractor()
    title_extractor = TitleExtractor()
    
    # Get all PDF files
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    pdf_files.sort()
    
    logger.info("=" * 80)
    logger.info("TITLE EXTRACTION ANALYSIS")
    logger.info("=" * 80)
    
    for pdf_file in pdf_files:
        expected_file = expected_outputs_dir / f"{pdf_file.stem}.json"
        
        if not expected_file.exists():
            continue
        
        logger.info(f"\nAnalyzing: {pdf_file.name}")
        logger.info("-" * 50)
        
        try:
            # Load expected title
            with open(expected_file, 'r', encoding='utf-8') as f:
                expected_data = json.load(f)
            expected_title = expected_data.get('title', '')
            
            # Extract text blocks and metadata
            text_blocks = pdf_extractor.extract_text_with_metadata(str(pdf_file))
            pdf_metadata = pdf_extractor.get_document_info(str(pdf_file))
            
            # Extract title using our current method
            extracted_title = title_extractor.extract_title(str(pdf_file), text_blocks, pdf_metadata)
            
            # Show comparison
            logger.info(f"Expected title: '{expected_title}'")
            logger.info(f"Extracted title: '{extracted_title}'")
            logger.info(f"Match: {expected_title.strip() == extracted_title.strip()}")
            
            # Show metadata for analysis
            if pdf_metadata:
                logger.info(f"PDF metadata title: '{pdf_metadata.get('title', 'N/A')}'")
            
            # Show first few text blocks for analysis
            logger.info("First 5 text blocks:")
            for i, block in enumerate(text_blocks[:5]):
                logger.info(f"  {i+1}. Page {block.page}: '{block.text}' (size: {block.font_size}, bold: {block.is_bold})")
            
            # Analyze what the expected title might be based on
            logger.info("\nAnalysis:")
            if expected_title.strip():
                # Check if expected title appears in text blocks
                found_in_blocks = False
                for i, block in enumerate(text_blocks[:10]):  # Check first 10 blocks
                    if expected_title.strip() in block.text or block.text in expected_title.strip():
                        logger.info(f"  Expected title found in block {i+1}: '{block.text}'")
                        found_in_blocks = True
                        break
                
                if not found_in_blocks:
                    logger.info("  Expected title not found in first 10 text blocks")
                    # Check if it's a combination or derived from multiple blocks
                    combined_text = " ".join(block.text for block in text_blocks[:5])
                    if expected_title.strip() in combined_text:
                        logger.info("  Expected title might be derived from combined text")
            else:
                logger.info("  Expected title is empty")
            
        except Exception as e:
            logger.error(f"Error analyzing {pdf_file.name}: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


def create_title_mapping_strategy():
    """Create a strategy to map our extracted titles to expected titles."""
    
    # Based on the expected outputs, create mapping rules
    title_mapping_rules = {
        "file01.pdf": {
            "expected": "Application form for grant of LTC advance  ",
            "strategy": "Look for form title in first few text blocks, not metadata"
        },
        "file02.pdf": {
            "expected": "Overview  Foundation Level Extensions  ",
            "strategy": "Combine 'Overview' + 'Foundation Level Extensions' from document"
        },
        "file03.pdf": {
            "expected": "RFP:Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library  ",
            "strategy": "Look for RFP title in document text, not just metadata"
        },
        "file04.pdf": {
            "expected": "Parsippany -Troy Hills STEM Pathways",
            "strategy": "Use metadata title as-is"
        },
        "file05.pdf": {
            "expected": "",
            "strategy": "Return empty string for this document type"
        }
    }
    
    logger.info("Title Mapping Strategy:")
    for pdf_name, info in title_mapping_rules.items():
        logger.info(f"{pdf_name}:")
        logger.info(f"  Expected: '{info['expected']}'")
        logger.info(f"  Strategy: {info['strategy']}")
        logger.info()


if __name__ == "__main__":
    analyze_title_extraction()
    create_title_mapping_strategy()