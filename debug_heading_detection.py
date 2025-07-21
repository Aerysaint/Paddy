#!/usr/bin/env python3
"""
Debug script to analyze heading detection issues in file03 and file04.
"""

import sys
from pathlib import Path
from src.pdf_extractor import PDFExtractor
from src.title_extractor import TitleExtractor
from src.structure_analyzer import StructureAnalyzer
from src.heading_level_classifier import HeadingLevelClassifier
from src.semantic_analyzer import SemanticAnalyzer
from src.confidence_fusion import ConfidenceFusionSystem

def debug_file03_title_issue():
    """Debug the title corruption issue in file03."""
    print("=" * 60)
    print("DEBUGGING FILE03 TITLE CORRUPTION")
    print("=" * 60)
    
    # Extract text blocks
    extractor = PDFExtractor()
    text_blocks = extractor.extract_text_with_metadata("pdfs/file03.pdf")
    metadata = None  # PDF metadata extraction would need separate method
    
    print(f"Extracted {len(text_blocks)} text blocks")
    
    # Debug title extraction
    title_extractor = TitleExtractor()
    
    # Get detailed title extraction info
    title_info = title_extractor.get_title_extraction_info("pdfs/file03.pdf", text_blocks, metadata)
    
    print(f"\nTitle extraction results:")
    print(f"Final title: '{title_info['extracted_title']}'")
    print(f"Method: {title_info['extraction_method']}")
    print(f"Metadata title: '{title_info['metadata_title']}'")
    print(f"Filename title: '{title_info['filename_title']}'")
    
    print(f"\nTop visual prominence candidates:")
    for i, candidate in enumerate(title_info['candidates'][:5]):
        print(f"  {i+1}. '{candidate['text']}' (score: {candidate['score']:.3f})")
    
    # Look at first page blocks specifically
    print(f"\nFirst page text blocks (first 10):")
    first_page_blocks = [b for b in text_blocks if b.page == 0][:10]
    for i, block in enumerate(first_page_blocks):
        print(f"  {i+1}. '{block.text}' (font: {block.font_size}, bold: {block.is_bold})")

def debug_file04_heading_detection():
    """Debug the missing PATHWAY OPTIONS heading in file04."""
    print("=" * 60)
    print("DEBUGGING FILE04 HEADING DETECTION")
    print("=" * 60)
    
    # Extract text blocks
    extractor = PDFExtractor()
    text_blocks = extractor.extract_text_with_metadata("pdfs/file04.pdf")
    metadata = None
    
    print(f"Extracted {len(text_blocks)} text blocks")
    
    # Look for "PATHWAY OPTIONS" specifically
    pathway_blocks = []
    for i, block in enumerate(text_blocks):
        if "pathway" in block.text.lower() or "option" in block.text.lower():
            pathway_blocks.append((i, block))
    
    print(f"\nBlocks containing 'pathway' or 'option':")
    for i, block in pathway_blocks:
        print(f"  Block {i}: '{block.text}' (font: {block.font_size}, bold: {block.is_bold})")
        print(f"    Position: {block.bbox}, Page: {block.page}")
    
    # Analyze structure
    analyzer = StructureAnalyzer()
    candidates, context = analyzer.analyze_document_structure(text_blocks)
    
    print(f"\nStructure analysis found {len(candidates)} candidates:")
    for i, candidate in enumerate(candidates):
        print(f"  {i+1}. '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
        if "pathway" in candidate.text.lower() or "option" in candidate.text.lower():
            print(f"      *** PATHWAY/OPTION CANDIDATE ***")
            print(f"      Features: {candidate.formatting_features}")
    
    # Check semantic analysis
    semantic_analyzer = SemanticAnalyzer()
    ambiguous_candidates = [c for c in candidates if 0.35 <= c.confidence_score < 0.75]
    
    if ambiguous_candidates:
        print(f"\nSemantic analysis on {len(ambiguous_candidates)} ambiguous candidates:")
        enhanced_candidates = semantic_analyzer.analyze_ambiguous_candidates(ambiguous_candidates, text_blocks)
        
        for candidate in enhanced_candidates:
            if "pathway" in candidate.text.lower() or "option" in candidate.text.lower():
                print(f"  PATHWAY/OPTION after semantic: '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
    
    # Check confidence fusion
    fusion = ConfidenceFusionSystem()
    fusion_result = fusion.analyze_document_with_fusion(text_blocks)
    final_candidates = fusion_result.candidates
    
    print(f"\nAfter confidence fusion ({len(final_candidates)} candidates):")
    for candidate in final_candidates:
        if "pathway" in candidate.text.lower() or "option" in candidate.text.lower():
            print(f"  PATHWAY/OPTION final: '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
    
    # Check level classification
    classifier = HeadingLevelClassifier()
    classified = classifier.classify_heading_levels(final_candidates)
    
    print(f"\nAfter level classification:")
    for candidate in classified:
        if "pathway" in candidate.text.lower() or "option" in candidate.text.lower():
            level = candidate.formatting_features.get('assigned_level', 'unknown')
            print(f"  PATHWAY/OPTION classified: '{candidate.text}' (level: H{level}, confidence: {candidate.confidence_score:.3f})")

def debug_all_file04_text():
    """Show all text blocks from file04 to find PATHWAY OPTIONS."""
    print("=" * 60)
    print("ALL FILE04 TEXT BLOCKS")
    print("=" * 60)
    
    extractor = PDFExtractor()
    text_blocks = extractor.extract_text_with_metadata("pdfs/file04.pdf")
    metadata = None
    
    for i, block in enumerate(text_blocks):
        print(f"Block {i:2d}: '{block.text}' (font: {block.font_size:.1f}, bold: {block.is_bold})")
        if "pathway" in block.text.lower():
            print("    *** CONTAINS PATHWAY ***")

if __name__ == "__main__":
    debug_file03_title_issue()
    print("\n")
    debug_file04_heading_detection()
    print("\n")
    debug_all_file04_text()