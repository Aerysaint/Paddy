#!/usr/bin/env python3
"""
Debug script to analyze file05 issues.
"""

import sys
from pathlib import Path
from src.pdf_extractor import PDFExtractor
from src.title_extractor import TitleExtractor
from src.structure_analyzer import StructureAnalyzer
from src.heading_level_classifier import HeadingLevelClassifier
from src.semantic_analyzer import SemanticAnalyzer
from src.confidence_fusion import ConfidenceFusionSystem

def debug_file05():
    """Debug file05 issues."""
    print("=" * 60)
    print("DEBUGGING FILE05 - TOPJUMP PARTY INVITATION")
    print("=" * 60)
    
    # Extract text blocks
    extractor = PDFExtractor()
    text_blocks = extractor.extract_text_with_metadata("pdfs/file05.pdf")
    metadata = None
    
    print(f"Extracted {len(text_blocks)} text blocks")
    
    # Show all text blocks
    print(f"\nAll text blocks:")
    for i, block in enumerate(text_blocks):
        print(f"  Block {i:2d}: '{block.text}' (font: {block.font_size:.1f}, bold: {block.is_bold})")
        if "hope" in block.text.lower() or "party" in block.text.lower() or "you" in block.text.lower():
            print("    *** POTENTIAL HEADING ***")
    
    # Debug title extraction
    title_extractor = TitleExtractor()
    title_info = title_extractor.get_title_extraction_info("pdfs/file05.pdf", text_blocks, metadata)
    
    print(f"\nTitle extraction results:")
    print(f"Final title: '{title_info['extracted_title']}'")
    print(f"Method: {title_info['extraction_method']}")
    
    print(f"\nTop visual prominence candidates:")
    for i, candidate in enumerate(title_info['candidates'][:5]):
        print(f"  {i+1}. '{candidate['text']}' (score: {candidate['score']:.3f})")
    
    # Analyze structure
    analyzer = StructureAnalyzer()
    candidates, context = analyzer.analyze_document_structure(text_blocks)
    
    print(f"\nStructure analysis found {len(candidates)} candidates:")
    for i, candidate in enumerate(candidates):
        print(f"  {i+1}. '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
        if "hope" in candidate.text.lower() or "party" in candidate.text.lower():
            print(f"      *** TARGET HEADING CANDIDATE ***")
            print(f"      Features: {candidate.formatting_features}")
    
    # Check semantic analysis
    semantic_analyzer = SemanticAnalyzer()
    ambiguous_candidates = [c for c in candidates if 0.30 <= c.confidence_score < 0.75]
    
    if ambiguous_candidates:
        print(f"\nSemantic analysis on {len(ambiguous_candidates)} ambiguous candidates:")
        enhanced_candidates = semantic_analyzer.analyze_ambiguous_candidates(ambiguous_candidates, text_blocks)
        
        for candidate in enhanced_candidates:
            print(f"  After semantic: '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
            if "hope" in candidate.text.lower() or "party" in candidate.text.lower():
                print(f"      *** TARGET HEADING AFTER SEMANTIC ***")
    
    # Check confidence fusion
    fusion = ConfidenceFusionSystem()
    fusion_result = fusion.analyze_document_with_fusion(text_blocks)
    final_candidates = fusion_result.candidates
    
    print(f"\nAfter confidence fusion ({len(final_candidates)} candidates):")
    for candidate in final_candidates:
        print(f"  Final: '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
        if "hope" in candidate.text.lower() or "party" in candidate.text.lower():
            print(f"      *** TARGET HEADING FINAL ***")
    
    # Check level classification
    classifier = HeadingLevelClassifier()
    classified = classifier.classify_heading_levels(final_candidates)
    
    print(f"\nAfter level classification:")
    for candidate in classified:
        level = candidate.formatting_features.get('assigned_level', 'unknown')
        print(f"  Classified: '{candidate.text}' (level: H{level}, confidence: {candidate.confidence_score:.3f})")
        if "hope" in candidate.text.lower() or "party" in candidate.text.lower():
            print(f"      *** TARGET HEADING CLASSIFIED ***")
    
    # Check final filtering (simulate batch processor logic)
    print(f"\nFinal filtering simulation:")
    base_confidence_threshold = 0.30  # Updated to match batch processor
    numbered_confidence_threshold = 0.3
    
    final_headings = []
    for candidate in classified:
        has_numbering = any('numbering_level_' in indicator for indicator in candidate.level_indicators)
        confidence_threshold = numbered_confidence_threshold if has_numbering else base_confidence_threshold
        
        if candidate.assigned_level and candidate.confidence_score >= confidence_threshold:
            final_headings.append(candidate)
            print(f"  PASSED: '{candidate.text}' (confidence: {candidate.confidence_score:.3f} >= {confidence_threshold:.2f})")
        else:
            print(f"  FILTERED: '{candidate.text}' (confidence: {candidate.confidence_score:.3f} < {confidence_threshold:.2f})")
    
    print(f"\nFinal headings that would appear in output: {len(final_headings)}")
    for heading in final_headings:
        level = heading.formatting_features.get('assigned_level', 'unknown')
        print(f"  H{level}: '{heading.text}'")

if __name__ == "__main__":
    debug_file05()