"""
Test the confidence fusion system on actual PDFs, store results separately, and compare with expected outputs.
"""

import json
import time
import logging
import os
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
    """Convert numeric level to heading format (H1, H2, H3, H4)."""
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
        
        # Step 5: Filter and convert to output format
        logger.debug("Converting to output format...")
        outline = []
        
        # Sort candidates by page and position
        sorted_candidates = sorted(
            classified_candidates, 
            key=lambda c: (c.page, c.text_block.bbox[1] if c.text_block else 0)
        )
        
        for candidate in sorted_candidates:
            # Use a confidence threshold to filter results
            if candidate.confidence_score >= 0.4:  # Adjusted threshold
                level = candidate.formatting_features.get('assigned_level', 1)
                # Ensure level is within valid range (1-4 to match expected outputs)
                level = max(1, min(4, level))
                
                outline.append({
                    "level": convert_level_to_heading_format(level),
                    "text": candidate.text.strip(),
                    "page": candidate.page
                })
        
        logger.info(f"Generated outline with {len(outline)} headings")
        
        return {
            "title": title.strip(),
            "outline": outline
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"title": "", "outline": []}


def save_result_to_file(result: Dict[str, Any], output_path: str) -> None:
    """Save result to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved result to {output_path}")
    except Exception as e:
        logger.error(f"Error saving result to {output_path}: {str(e)}")


def load_expected_output(output_path: str) -> Dict[str, Any]:
    """Load expected output from JSON file."""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading expected output {output_path}: {str(e)}")
        return {"title": "", "outline": []}


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace."""
    return ' '.join(text.strip().split())


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
        'title_match': False,
        'actual_headings': len(actual.get('outline', [])),
        'expected_headings': len(expected.get('outline', [])),
        'heading_matches': 0,
        'text_matches': 0,
        'level_matches': 0,
        'page_matches': 0,
        'partial_text_matches': 0
    }
    
    # Compare titles (normalize whitespace)
    actual_title = normalize_text(actual.get('title', ''))
    expected_title = normalize_text(expected.get('title', ''))
    comparison['title_match'] = actual_title == expected_title
    
    actual_outline = actual.get('outline', [])
    expected_outline = expected.get('outline', [])
    
    # Create a mapping of expected headings for better matching
    expected_texts = [normalize_text(h.get('text', '')) for h in expected_outline]
    
    # Compare headings
    for actual_heading in actual_outline:
        actual_text = normalize_text(actual_heading.get('text', ''))
        actual_level = actual_heading.get('level', '')
        actual_page = actual_heading.get('page', 0)
        
        # Find best match in expected headings
        best_match_idx = -1
        exact_match = False
        
        for i, expected_heading in enumerate(expected_outline):
            expected_text = normalize_text(expected_heading.get('text', ''))
            
            if actual_text == expected_text:
                best_match_idx = i
                exact_match = True
                break
            elif actual_text in expected_text or expected_text in actual_text:
                if best_match_idx == -1:  # Only set if no exact match found yet
                    best_match_idx = i
        
        if best_match_idx >= 0:
            expected_heading = expected_outline[best_match_idx]
            expected_text = normalize_text(expected_heading.get('text', ''))
            expected_level = expected_heading.get('level', '')
            expected_page = expected_heading.get('page', 0)
            
            if exact_match:
                comparison['text_matches'] += 1
            else:
                comparison['partial_text_matches'] += 1
            
            if actual_level == expected_level:
                comparison['level_matches'] += 1
            
            if actual_page == expected_page:
                comparison['page_matches'] += 1
            
            if exact_match and actual_level == expected_level and actual_page == expected_page:
                comparison['heading_matches'] += 1
    
    # Calculate accuracy metrics
    if comparison['expected_headings'] > 0:
        comparison['text_accuracy'] = comparison['text_matches'] / comparison['expected_headings']
        comparison['partial_text_accuracy'] = (comparison['text_matches'] + comparison['partial_text_matches']) / comparison['expected_headings']
        comparison['level_accuracy'] = comparison['level_matches'] / comparison['expected_headings']
        comparison['page_accuracy'] = comparison['page_matches'] / comparison['expected_headings']
        comparison['overall_accuracy'] = comparison['heading_matches'] / comparison['expected_headings']
    else:
        comparison['text_accuracy'] = 1.0 if comparison['actual_headings'] == 0 else 0.0
        comparison['partial_text_accuracy'] = 1.0 if comparison['actual_headings'] == 0 else 0.0
        comparison['level_accuracy'] = 1.0 if comparison['actual_headings'] == 0 else 0.0
        comparison['page_accuracy'] = 1.0 if comparison['actual_headings'] == 0 else 0.0
        comparison['overall_accuracy'] = 1.0 if comparison['actual_headings'] == 0 else 0.0
    
    return comparison


def test_fusion_with_separate_output():
    """Test the confidence fusion system and store results in separate folder."""
    
    pdfs_dir = Path("pdfs")
    expected_outputs_dir = Path("outputs")
    actual_outputs_dir = Path("outputs_fusion")
    
    # Create output directory for our results
    actual_outputs_dir.mkdir(exist_ok=True)
    
    if not pdfs_dir.exists():
        logger.error("PDFs directory not found!")
        return
    
    if not expected_outputs_dir.exists():
        logger.error("Expected outputs directory not found!")
        return
    
    # Get all PDF files
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    pdf_files.sort()
    
    logger.info(f"Found {len(pdf_files)} PDF files to test")
    logger.info(f"Results will be saved to: {actual_outputs_dir}")
    
    results = []
    total_start_time = time.time()
    
    for pdf_file in pdf_files:
        logger.info("=" * 60)
        logger.info(f"Testing: {pdf_file.name}")
        logger.info("=" * 60)
        
        # Expected output file
        expected_file = expected_outputs_dir / f"{pdf_file.stem}.json"
        actual_file = actual_outputs_dir / f"{pdf_file.stem}.json"
        
        if not expected_file.exists():
            logger.warning(f"Expected output file not found: {expected_file}")
            continue
        
        # Process PDF with fusion system
        start_time = time.time()
        actual_result = process_pdf_with_fusion(str(pdf_file))
        processing_time = time.time() - start_time
        
        # Save our result
        save_result_to_file(actual_result, str(actual_file))
        
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
        logger.info(f"  Exact text accuracy: {comparison['text_accuracy']:.2%}")
        logger.info(f"  Partial text accuracy: {comparison['partial_text_accuracy']:.2%}")
        logger.info(f"  Level accuracy: {comparison['level_accuracy']:.2%}")
        logger.info(f"  Page accuracy: {comparison['page_accuracy']:.2%}")
        logger.info(f"  Overall accuracy: {comparison['overall_accuracy']:.2%}")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        logger.info(f"  Saved to: {actual_file}")
        
        # Show some actual vs expected headings for debugging
        if comparison['actual_headings'] > 0 and comparison['expected_headings'] > 0:
            logger.info("  Sample headings (actual vs expected):")
            actual_outline = actual_result.get('outline', [])
            expected_outline = expected_result.get('outline', [])
            
            for i in range(min(3, len(actual_outline), len(expected_outline))):
                actual_h = actual_outline[i]
                expected_h = expected_outline[i]
                actual_text = normalize_text(actual_h.get('text', ''))
                expected_text = normalize_text(expected_h.get('text', ''))
                match_indicator = "✓" if (actual_text == expected_text and 
                                        actual_h.get('level') == expected_h.get('level')) else "✗"
                logger.info(f"    {match_indicator} '{actual_text}' ({actual_h.get('level', '')}) vs '{expected_text}' ({expected_h.get('level', '')})")
    
    # Calculate overall statistics
    total_time = time.time() - total_start_time
    
    if results:
        avg_title_match = sum(1 for r in results if r['title_match']) / len(results)
        avg_text_accuracy = sum(r['text_accuracy'] for r in results) / len(results)
        avg_partial_text_accuracy = sum(r['partial_text_accuracy'] for r in results) / len(results)
        avg_level_accuracy = sum(r['level_accuracy'] for r in results) / len(results)
        avg_page_accuracy = sum(r['page_accuracy'] for r in results) / len(results)
        avg_overall_accuracy = sum(r['overall_accuracy'] for r in results) / len(results)
        avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
        
        logger.info("=" * 60)
        logger.info("OVERALL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Files tested: {len(results)}")
        logger.info(f"Title match rate: {avg_title_match:.2%}")
        logger.info(f"Average exact text accuracy: {avg_text_accuracy:.2%}")
        logger.info(f"Average partial text accuracy: {avg_partial_text_accuracy:.2%}")
        logger.info(f"Average level accuracy: {avg_level_accuracy:.2%}")
        logger.info(f"Average page accuracy: {avg_page_accuracy:.2%}")
        logger.info(f"Average overall accuracy: {avg_overall_accuracy:.2%}")
        logger.info(f"Average processing time: {avg_processing_time:.2f}s")
        logger.info(f"Total testing time: {total_time:.2f}s")
        logger.info(f"Results saved to: {actual_outputs_dir}")
        
        # Show per-file results
        logger.info("\nPer-file results:")
        for result in results:
            logger.info(f"  {result['pdf_name']}: {result['overall_accuracy']:.2%} accuracy, {result['processing_time']:.2f}s")
        
        # Save summary results
        summary_file = actual_outputs_dir / "test_summary.json"
        summary = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files_tested": len(results),
            "total_time": total_time,
            "average_metrics": {
                "title_match_rate": avg_title_match,
                "exact_text_accuracy": avg_text_accuracy,
                "partial_text_accuracy": avg_partial_text_accuracy,
                "level_accuracy": avg_level_accuracy,
                "page_accuracy": avg_page_accuracy,
                "overall_accuracy": avg_overall_accuracy,
                "processing_time": avg_processing_time
            },
            "per_file_results": results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test summary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    test_fusion_with_separate_output()