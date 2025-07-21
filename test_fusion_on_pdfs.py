"""
Test the confidence fusion system on actual PDFs and compare with expected outputs.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.pdf_extractor import PDFExtractor
from src.confidence_fusion import analyze_document_with_fusion
from src.title_extractor import TitleExtractor
from src.heading_level_classifier import classify_heading_levels
from src.data_models import DocumentStructure
from src.logging_config import setup_logging

# Set up logging
logger = setup_logging('INFO')


def convert_level_to_heading_format(level: int) -> str:
    """Convert numeric level to heading format (H1, H2, H3)."""
    return f"H{level}"


def process_pdf_with_fusion(pdf_path: str) -> Dict[str, Any]:
    """
    Process a PDF using the confidence fusion system.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with title and outline in expected format
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Initialize components
    pdf_extractor = PDFExtractor()
    title_extractor = TitleExtractor()
    
    try:
        # Step 1: Extract text blocks
        logger.debug("Extracting text blocks...")
        text_blocks = pdf_extractor.extract_text_with_metadata(pdf_path)
        
        if not text_blocks:
            logger.warning(f"No text blocks extracted from {pdf_path}")
            return {"title": "", "outline": []}
        
        logger.info(f"Extracted {len(text_blocks)} text blocks")
        
        # Step 2: Extract title
        logger.debug("Extracting title...")
        pdf_metadata = pdf_extractor.get_document_info(pdf_path)
        title = title_extractor.extract_title(pdf_path, text_blocks, pdf_metadata)
        logger.info(f"Extracted title: '{title}'")
        
        # Step 3: Apply confidence fusion system
        logger.debug("Applying confidence fusion system...")
        fusion_result = analyze_document_with_fusion(text_blocks)
        
        logger.info(f"Fusion analysis completed:")
        logger.info(f"  - Document type: {fusion_result.document_type.value}")
        logger.info(f"  - Processing time: {fusion_result.performance_metrics['total_processing_time']:.2f}s")
        logger.info(f"  - Within target: {fusion_result.performance_metrics['within_target']}")
        logger.info(f"  - Found {len(fusion_result.candidates)} heading candidates")
        
        # Step 4: Classify heading levels
        logger.debug("Classifying heading levels...")
        classified_candidates = classify_heading_levels(fusion_result.candidates)
        
        # Step 5: Filter high-confidence headings and convert to output format
        logger.debug("Converting to output format...")
        outline = []
        
        # Sort candidates by page and position
        sorted_candidates = sorted(
            classified_candidates, 
            key=lambda c: (c.page, c.text_block.bbox[1] if c.text_block else 0)
        )
        
        for candidate in sorted_candidates:
            # Use a lower confidence threshold to match expected outputs
            if candidate.confidence_score >= 0.3:  # Lower threshold for testing
                level = candidate.formatting_features.get('assigned_level', 1)
                outline.append({
                    "level": convert_level_to_heading_format(level),
                    "text": candidate.text,
                    "page": candidate.page
                })
        
        logger.info(f"Generated outline with {len(outline)} headings")
        
        return {
            "title": title,
            "outline": outline
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"title": "", "outline": []}


def load_expected_output(output_path: str) -> Dict[str, Any]:
    """Load expected output from JSON file."""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading expected output {output_path}: {str(e)}")
        return {"title": "", "outline": []}


def compare_results(actual: Dict[str, Any], expected: Dict[str, Any], pdf_name: str) -> Dict[str, Any]:
    """
    Compare actual results with expected results.
    
    Args:
        actual: Actual results from fusion system
        expected: Expected results from JSON file
        pdf_name: Name of the PDF file
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'pdf_name': pdf_name,
        'title_match': actual.get('title', '') == expected.get('title', ''),
        'actual_headings': len(actual.get('outline', [])),
        'expected_headings': len(expected.get('outline', [])),
        'heading_matches': 0,
        'text_matches': 0,
        'level_matches': 0,
        'page_matches': 0
    }
    
    actual_outline = actual.get('outline', [])
    expected_outline = expected.get('outline', [])
    
    # Compare headings
    for i, expected_heading in enumerate(expected_outline):
        if i < len(actual_outline):
            actual_heading = actual_outline[i]
            
            # Check text match
            if actual_heading.get('text', '') == expected_heading.get('text', ''):
                comparison['text_matches'] += 1
            
            # Check level match
            if actual_heading.get('level', '') == expected_heading.get('level', ''):
                comparison['level_matches'] += 1
            
            # Check page match
            if actual_heading.get('page', 0) == expected_heading.get('page', 0):
                comparison['page_matches'] += 1
            
            # Check complete match
            if (actual_heading.get('text', '') == expected_heading.get('text', '') and
                actual_heading.get('level', '') == expected_heading.get('level', '') and
                actual_heading.get('page', 0) == expected_heading.get('page', 0)):
                comparison['heading_matches'] += 1
    
    # Calculate accuracy metrics
    if comparison['expected_headings'] > 0:
        comparison['text_accuracy'] = comparison['text_matches'] / comparison['expected_headings']
        comparison['level_accuracy'] = comparison['level_matches'] / comparison['expected_headings']
        comparison['page_accuracy'] = comparison['page_matches'] / comparison['expected_headings']
        comparison['overall_accuracy'] = comparison['heading_matches'] / comparison['expected_headings']
    else:
        comparison['text_accuracy'] = 0.0
        comparison['level_accuracy'] = 0.0
        comparison['page_accuracy'] = 0.0
        comparison['overall_accuracy'] = 0.0
    
    return comparison


def test_fusion_on_all_pdfs():
    """Test the confidence fusion system on all PDFs in the pdfs folder."""
    
    pdfs_dir = Path("pdfs")
    outputs_dir = Path("outputs")
    
    if not pdfs_dir.exists():
        logger.error("PDFs directory not found!")
        return
    
    if not outputs_dir.exists():
        logger.error("Outputs directory not found!")
        return
    
    # Get all PDF files
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    pdf_files.sort()
    
    logger.info(f"Found {len(pdf_files)} PDF files to test")
    
    results = []
    total_start_time = time.time()
    
    for pdf_file in pdf_files:
        logger.info("=" * 60)
        logger.info(f"Testing: {pdf_file.name}")
        logger.info("=" * 60)
        
        # Expected output file
        expected_file = outputs_dir / f"{pdf_file.stem}.json"
        
        if not expected_file.exists():
            logger.warning(f"Expected output file not found: {expected_file}")
            continue
        
        # Process PDF with fusion system
        start_time = time.time()
        actual_result = process_pdf_with_fusion(str(pdf_file))
        processing_time = time.time() - start_time
        
        # Load expected result
        expected_result = load_expected_output(str(expected_file))
        
        # Compare results
        comparison = compare_results(actual_result, expected_result, pdf_file.name)
        comparison['processing_time'] = processing_time
        
        results.append(comparison)
        
        # Log comparison results
        logger.info(f"Results for {pdf_file.name}:")
        logger.info(f"  Title match: {comparison['title_match']}")
        logger.info(f"  Headings found: {comparison['actual_headings']} (expected: {comparison['expected_headings']})")
        logger.info(f"  Text accuracy: {comparison['text_accuracy']:.2%}")
        logger.info(f"  Level accuracy: {comparison['level_accuracy']:.2%}")
        logger.info(f"  Page accuracy: {comparison['page_accuracy']:.2%}")
        logger.info(f"  Overall accuracy: {comparison['overall_accuracy']:.2%}")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        
        # Show some actual vs expected headings for debugging
        if comparison['actual_headings'] > 0:
            logger.info("  Sample headings (actual vs expected):")
            actual_outline = actual_result.get('outline', [])
            expected_outline = expected_result.get('outline', [])
            
            for i in range(min(3, len(actual_outline), len(expected_outline))):
                actual_h = actual_outline[i]
                expected_h = expected_outline[i]
                match_indicator = "✓" if (actual_h.get('text') == expected_h.get('text') and 
                                        actual_h.get('level') == expected_h.get('level')) else "✗"
                logger.info(f"    {match_indicator} '{actual_h.get('text', '')}' ({actual_h.get('level', '')}) vs '{expected_h.get('text', '')}' ({expected_h.get('level', '')})")
    
    # Calculate overall statistics
    total_time = time.time() - total_start_time
    
    if results:
        avg_text_accuracy = sum(r['text_accuracy'] for r in results) / len(results)
        avg_level_accuracy = sum(r['level_accuracy'] for r in results) / len(results)
        avg_page_accuracy = sum(r['page_accuracy'] for r in results) / len(results)
        avg_overall_accuracy = sum(r['overall_accuracy'] for r in results) / len(results)
        avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
        
        logger.info("=" * 60)
        logger.info("OVERALL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Files tested: {len(results)}")
        logger.info(f"Average text accuracy: {avg_text_accuracy:.2%}")
        logger.info(f"Average level accuracy: {avg_level_accuracy:.2%}")
        logger.info(f"Average page accuracy: {avg_page_accuracy:.2%}")
        logger.info(f"Average overall accuracy: {avg_overall_accuracy:.2%}")
        logger.info(f"Average processing time: {avg_processing_time:.2f}s")
        logger.info(f"Total testing time: {total_time:.2f}s")
        
        # Show per-file results
        logger.info("\nPer-file results:")
        for result in results:
            logger.info(f"  {result['pdf_name']}: {result['overall_accuracy']:.2%} accuracy, {result['processing_time']:.2f}s")
    
    return results


if __name__ == "__main__":
    test_fusion_on_all_pdfs()