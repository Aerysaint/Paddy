#!/usr/bin/env python3
"""
Demonstration of JSON handler functionality.
"""

import json
from src.json_handler import JSONHandler
from src.data_models import DocumentStructure, HeadingCandidate


def demo_json_handler():
    """Demonstrate JSON handler capabilities."""
    
    print("=== PDF Outline Extractor - JSON Handler Demo ===\n")
    
    # Initialize handler
    handler = JSONHandler()
    print("✓ JSON Handler initialized with schema validation")
    
    # Demo 1: Basic usage with multilingual content
    print("\n1. Basic JSON Output with Multilingual Content")
    print("-" * 50)
    
    title = "Research Paper: Machine Learning Applications in 日本語 Processing"
    headings = [
        {"level": 1, "text": "Abstract", "page": 1},
        {"level": 1, "text": "Introduction", "page": 2},
        {"level": 2, "text": "Background & Motivation", "page": 3},
        {"level": 2, "text": "Research Questions", "page": 4},
        {"level": 1, "text": "Literature Review", "page": 5},
        {"level": 2, "text": "Natural Language Processing", "page": 6},
        {"level": 3, "text": "Japanese Text Analysis", "page": 7},
        {"level": 3, "text": "Mathematical Models: ∑(xi) ≈ ∫f(x)dx", "page": 8},
        {"level": 1, "text": "Methodology", "page": 10},
        {"level": 2, "text": "Data Collection", "page": 11},
        {"level": 2, "text": "Experimental Setup", "page": 13},
        {"level": 1, "text": "Results & Discussion", "page": 15},
        {"level": 2, "text": "Performance Metrics", "page": 16},
        {"level": 2, "text": "Statistical Analysis", "page": 18},
        {"level": 1, "text": "Conclusion", "page": 20},
        {"level": 1, "text": "References", "page": 21}
    ]
    
    # Create JSON output
    json_output = handler.create_json_output(title, headings, validate=True)
    
    print(f"Title: {json_output['title']}")
    print(f"Number of headings: {len(json_output['outline'])}")
    print("\nHeading structure:")
    
    for i, heading in enumerate(json_output['outline'][:8]):  # Show first 8
        indent = "  " * (int(heading['level'][1]) - 1)
        print(f"{indent}{heading['level']}: {heading['text']} (page {heading['page']})")
    
    if len(json_output['outline']) > 8:
        print(f"  ... and {len(json_output['outline']) - 8} more headings")
    
    # Demo 2: Edge case handling
    print("\n\n2. Edge Case Handling")
    print("-" * 30)
    
    problematic_headings = [
        {"level": 1, "text": "Valid Heading", "page": 1},
        {"level": "invalid", "text": "", "page": 2},  # Empty text
        {"text": "Missing Level", "page": 3},  # Missing level
        "not a dictionary",  # Invalid format
        {"level": 2, "text": "Another Valid Heading", "page": 4},
        {"level": 3, "text": "Special chars: café, naïve, résumé", "page": 5}
    ]
    
    edge_case_output = handler.create_json_output("Test Document", problematic_headings)
    
    print(f"Input headings: {len(problematic_headings)}")
    print(f"Valid output headings: {len(edge_case_output['outline'])}")
    print("Filtered headings:")
    
    for heading in edge_case_output['outline']:
        print(f"  {heading['level']}: {heading['text']} (page {heading['page']})")
    
    # Demo 3: DocumentStructure integration
    print("\n\n3. DocumentStructure Integration")
    print("-" * 35)
    
    doc_structure = DocumentStructure(
        title="Technical Manual: System Architecture",
        headings=[
            {"level": 1, "text": "Overview", "page": 1},
            {"level": 2, "text": "System Requirements", "page": 2},
            {"level": 2, "text": "Architecture Design", "page": 4},
            {"level": 3, "text": "Database Layer", "page": 5},
            {"level": 3, "text": "API Layer", "page": 7},
            {"level": 1, "text": "Implementation", "page": 10}
        ],
        metadata={"pages": 15, "version": "1.0"}
    )
    
    structure_output = handler.format_from_document_structure(doc_structure)
    
    print(f"Document: {structure_output['title']}")
    print("Structure:")
    for heading in structure_output['outline']:
        indent = "  " * (int(heading['level'][1]) - 1)
        print(f"{indent}{heading['level']}: {heading['text']} (page {heading['page']})")
    
    # Demo 4: Schema validation
    print("\n\n4. Schema Validation")
    print("-" * 20)
    
    # Valid data
    valid_data = {"title": "Test", "outline": [{"level": "H1", "text": "Test", "page": 1}]}
    valid_result = handler.validate_schema(valid_data)
    print(f"Valid data validation: {'✓ PASS' if valid_result else '✗ FAIL'}")
    
    # Invalid data (missing required field)
    invalid_data = {"title": "Test", "outline": [{"level": "H1", "text": "Test"}]}  # Missing page
    invalid_result = handler.validate_schema(invalid_data)
    print(f"Invalid data validation: {'✗ FAIL (expected)' if not invalid_result else '✓ UNEXPECTED PASS'}")
    
    # Demo 5: File output
    print("\n\n5. File Output")
    print("-" * 15)
    
    output_file = "demo_output.json"
    success = handler.write_json_file(json_output, output_file)
    
    if success:
        print(f"✓ JSON output written to {output_file}")
        
        # Show file size and encoding
        import os
        file_size = os.path.getsize(output_file)
        print(f"  File size: {file_size} bytes")
        
        # Verify multilingual content in file
        with open(output_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
            
        multilingual_preserved = "日本語" in file_content and "∑" in file_content
        print(f"  Multilingual content preserved: {'✓ YES' if multilingual_preserved else '✗ NO'}")
        
        # Show a snippet
        print(f"  File preview (first 200 chars):")
        print(f"  {file_content[:200]}...")
        
    else:
        print("✗ Failed to write JSON output")
    
    print("\n=== Demo Complete ===")
    print("The JSON handler successfully:")
    print("✓ Formats PDF outline data to match the required schema")
    print("✓ Preserves multilingual content and special characters")
    print("✓ Validates output against JSON schema")
    print("✓ Handles edge cases gracefully")
    print("✓ Integrates with DocumentStructure objects")
    print("✓ Writes UTF-8 encoded JSON files")


if __name__ == "__main__":
    demo_json_handler()