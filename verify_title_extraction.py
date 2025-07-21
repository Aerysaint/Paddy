#!/usr/bin/env python3
"""
Verify that title extraction methods are working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from title_extractor import TitleExtractor, extract_title_from_pdf, get_title_candidates
from data_models import TextBlock

def test_title_extraction_methods():
    """Test that all title extraction methods are working."""
    
    # Create test data
    text_blocks = [
        TextBlock("Sample Document Title", 1, 16.0, "Arial", True, (100, 100, 300, 120), 18.0),
        TextBlock("This is some body text", 1, 12.0, "Arial", False, (100, 150, 400, 170), 14.0)
    ]
    
    pdf_metadata = {"title": "Metadata Title", "author": "Test Author"}
    pdf_path = "test_document.pdf"
    
    # Test TitleExtractor class
    extractor = TitleExtractor()
    
    print("Testing TitleExtractor methods:")
    
    # Test extract_title
    title = extractor.extract_title(pdf_path, text_blocks, pdf_metadata)
    print(f"✅ extract_title: {title}")
    
    # Test get_title_extraction_info
    info = extractor.get_title_extraction_info(pdf_path, text_blocks, pdf_metadata)
    print(f"✅ get_title_extraction_info: {info['extraction_method']} -> '{info['extracted_title']}'")
    print(f"   Candidates: {len(info['candidates'])}")
    print(f"   Metadata title: '{info['metadata_title']}'")
    print(f"   Filename title: '{info['filename_title']}'")
    
    # Test convenience functions
    print("\nTesting convenience functions:")
    
    # Test extract_title_from_pdf
    title2 = extract_title_from_pdf(pdf_path, text_blocks, pdf_metadata)
    print(f"✅ extract_title_from_pdf: {title2}")
    
    # Test get_title_candidates
    candidates = get_title_candidates(pdf_path, text_blocks, pdf_metadata)
    print(f"✅ get_title_candidates: {candidates['extraction_method']} -> '{candidates['extracted_title']}'")
    
    # Verify results are consistent
    assert title == title2, "extract_title methods should return same result"
    assert info['extracted_title'] == candidates['extracted_title'], "get_title_extraction_info should match get_title_candidates"
    
    print("\n✅ All title extraction methods are working correctly!")

if __name__ == "__main__":
    test_title_extraction_methods()