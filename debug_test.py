#!/usr/bin/env python3

from src.pdf_extractor import PDFExtractor
from src.structure_analyzer import StructureAnalyzer
from src.heading_level_classifier import HeadingLevelClassifier

def debug_page_numbers_pipeline():
    """Debug page numbers through the pipeline"""
    extractor = PDFExtractor()
    analyzer = StructureAnalyzer()
    classifier = HeadingLevelClassifier()
    
    text_blocks = extractor.extract_text_with_metadata('pdfs/file04.pdf')
    candidates, context = analyzer.analyze_document_structure(text_blocks)
    
    print("Candidates from structure analyzer:")
    for candidate in candidates:
        if "PATHWAY" in candidate.text:
            print(f'- "{candidate.text}" (page: {candidate.page})')
    
    classified = classifier.classify_heading_levels(candidates)
    
    print("\nAfter level classification:")
    for candidate in classified:
        if "PATHWAY" in candidate.text:
            print(f'- "{candidate.text}" (page: {candidate.page})')

if __name__ == "__main__":
    debug_page_numbers_pipeline()