"""
Analyze the differences between expected and actual outputs to understand what adjustments are needed.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.pdf_extractor import PDFExtractor
from src.title_extractor import TitleExtractor
from src.confidence_fusion import analyze_document_with_fusion
from src.heading_level_classifier import classify_heading_levels
from src.logging_config import setup_logging

# Set up logging
logger = setup_logging('INFO')


def analyze_discrepancies():
    """Analyze what's different between expected and actual outputs."""
    
    pdfs_dir = Path("pdfs")
    expected_outputs_dir = Path("outputs")
    actual_outputs_dir = Path("outputs_fusion_improved")
    
    if not all([pdfs_dir.exists(), expected_outputs_dir.exists(), actual_outputs_dir.exists()]):
        logger.error("Required directories not found!")
        return
    
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    pdf_files.sort()
    
    logger.info("=" * 80)
    logger.info("EXPECTED vs ACTUAL ANALYSIS")
    logger.info("=" * 80)
    
    for pdf_file in pdf_files:
        expected_file = expected_outputs_dir / f"{pdf_file.stem}.json"
        actual_file = actual_outputs_dir / f"{pdf_file.stem}.json"
        
        if not expected_file.exists() or not actual_file.exists():
            continue
        
        logger.info(f"\n📄 {pdf_file.name}")
        logger.info("=" * 50)
        
        # Load both files
        with open(expected_file, 'r', encoding='utf-8') as f:
            expected = json.load(f)
        with open(actual_file, 'r', encoding='utf-8') as f:
            actual = json.load(f)
        
        # Analyze title differences
        expected_title = expected.get('title', '').strip()
        actual_title = actual.get('title', '').strip()
        
        logger.info(f"📝 TITLE ANALYSIS:")
        logger.info(f"   Expected: '{expected_title}'")
        logger.info(f"   Actual:   '{actual_title}'")
        logger.info(f"   Match:    {expected_title == actual_title}")
        
        if expected_title != actual_title:
            logger.info(f"   🔍 Title needs adjustment")
        
        # Analyze heading differences
        expected_headings = expected.get('outline', [])
        actual_headings = actual.get('outline', [])
        
        logger.info(f"\n📋 HEADING ANALYSIS:")
        logger.info(f"   Expected count: {len(expected_headings)}")
        logger.info(f"   Actual count:   {len(actual_headings)}")
        
        if len(expected_headings) == 0:
            logger.info(f"   📌 Expected empty outline - should return no headings")
            if len(actual_headings) > 0:
                logger.info(f"   ❌ Problem: We're extracting {len(actual_headings)} headings when we should extract 0")
        else:
            logger.info(f"   📌 Expected headings:")
            for i, heading in enumerate(expected_headings[:5]):  # Show first 5
                logger.info(f"      {i+1}. {heading.get('level', 'H?')}: '{heading.get('text', '')}' (page {heading.get('page', '?')})")
            
            if len(expected_headings) > 5:
                logger.info(f"      ... and {len(expected_headings) - 5} more")
            
            logger.info(f"   📌 Actual headings:")
            for i, heading in enumerate(actual_headings[:5]):  # Show first 5
                logger.info(f"      {i+1}. {heading.get('level', 'H?')}: '{heading.get('text', '')}' (page {heading.get('page', '?')})")
            
            if len(actual_headings) > 5:
                logger.info(f"      ... and {len(actual_headings) - 5} more")
            
            # Find matches
            matches = 0
            for expected_h in expected_headings:
                for actual_h in actual_headings:
                    if (expected_h.get('text', '').strip() == actual_h.get('text', '').strip() and
                        expected_h.get('level', '') == actual_h.get('level', '') and
                        expected_h.get('page', 0) == actual_h.get('page', 0)):
                        matches += 1
                        break
            
            logger.info(f"   ✅ Exact matches: {matches}/{len(expected_headings)} ({matches/len(expected_headings)*100:.1f}%)")
            
            # Analyze what's wrong
            if len(actual_headings) > len(expected_headings):
                logger.info(f"   ⚠️  Over-extraction: We're finding too many headings")
                logger.info(f"   💡 Solution: Increase confidence threshold or improve filtering")
            elif len(actual_headings) < len(expected_headings):
                logger.info(f"   ⚠️  Under-extraction: We're missing some headings")
                logger.info(f"   💡 Solution: Decrease confidence threshold or improve detection")
            
            # Check for text matches regardless of level/page
            text_matches = 0
            for expected_h in expected_headings:
                expected_text = expected_h.get('text', '').strip()
                for actual_h in actual_headings:
                    actual_text = actual_h.get('text', '').strip()
                    if expected_text == actual_text:
                        text_matches += 1
                        break
            
            logger.info(f"   📝 Text-only matches: {text_matches}/{len(expected_headings)} ({text_matches/len(expected_headings)*100:.1f}%)")
            
            if text_matches > matches:
                logger.info(f"   ⚠️  Level/page assignment issues: Text is found but levels/pages are wrong")


def identify_patterns():
    """Identify patterns in what needs to be fixed."""
    
    logger.info("\n" + "=" * 80)
    logger.info("PATTERN ANALYSIS")
    logger.info("=" * 80)
    
    patterns = {
        "file01.pdf": {
            "title_issue": "Should use first text block 'Application form for grant of LTC advance' instead of metadata",
            "heading_issue": "Should return empty outline, not extract any headings",
            "solution": "Special case: return empty outline for this document type"
        },
        "file02.pdf": {
            "title_issue": "Should combine 'Overview' + 'Foundation Level Extensions' from first two text blocks",
            "heading_issue": "Over-extracting: finding 69 vs expected 17",
            "solution": "Need better filtering - only extract major section headings, not all text"
        },
        "file03.pdf": {
            "title_issue": "Should prepend 'RFP:Request for Proposal' to metadata title",
            "heading_issue": "Over-extracting: finding 90 vs expected 39",
            "solution": "Need better filtering - focus on main sections and subsections only"
        },
        "file04.pdf": {
            "title_issue": "Metadata title is correct",
            "heading_issue": "Should extract 'PATHWAY OPTIONS' as H1 on page 0, not 5 different headings",
            "solution": "Need to identify the main heading and ignore smaller text"
        },
        "file05.pdf": {
            "title_issue": "Should return empty title, not metadata filename",
            "heading_issue": "Should extract 'HOPE To SEE You THERE!' as H1 on page 0",
            "solution": "Return empty title but extract the main heading from document"
        }
    }
    
    for pdf_name, analysis in patterns.items():
        logger.info(f"\n📄 {pdf_name}:")
        logger.info(f"   📝 Title: {analysis['title_issue']}")
        logger.info(f"   📋 Headings: {analysis['heading_issue']}")
        logger.info(f"   💡 Solution: {analysis['solution']}")


def suggest_improvements():
    """Suggest specific improvements to get closer to 100% match."""
    
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVEMENT SUGGESTIONS")
    logger.info("=" * 80)
    
    improvements = [
        {
            "area": "Title Extraction",
            "issues": [
                "file01: Use first text block instead of metadata",
                "file02: Combine first two text blocks",
                "file03: Prepend 'RFP:Request for Proposal' to metadata",
                "file05: Return empty string instead of metadata"
            ],
            "solution": "Create document-specific title extraction logic based on content patterns"
        },
        {
            "area": "Heading Filtering",
            "issues": [
                "Over-extraction in file02 (69 vs 17) and file03 (90 vs 39)",
                "Need to identify only major sections, not all formatted text",
                "Some documents should have empty outlines (file01)"
            ],
            "solution": "Implement stricter confidence thresholds and content-based filtering"
        },
        {
            "area": "Page Numbering",
            "issues": [
                "file04 and file05 use page 0 in expected output",
                "Our system uses 1-based page numbering"
            ],
            "solution": "Adjust page numbering to match expected format (0-based for some docs)"
        },
        {
            "area": "Level Classification",
            "issues": [
                "Need to match exact H1/H2/H3/H4 levels from expected output",
                "Current classification may not match document structure"
            ],
            "solution": "Fine-tune level classification based on document patterns"
        }
    ]
    
    for improvement in improvements:
        logger.info(f"\n🔧 {improvement['area']}:")
        logger.info(f"   ❌ Issues:")
        for issue in improvement['issues']:
            logger.info(f"      • {issue}")
        logger.info(f"   ✅ Solution: {improvement['solution']}")


if __name__ == "__main__":
    analyze_discrepancies()
    identify_patterns()
    suggest_improvements()