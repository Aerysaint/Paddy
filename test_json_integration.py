#!/usr/bin/env python3
"""
Integration test for JSON handler with actual schema validation.
"""

import json
import tempfile
from pathlib import Path

from src.json_handler import JSONHandler
from src.data_models import DocumentStructure


def test_json_handler_integration():
    """Test JSON handler with real schema and multilingual content."""
    
    # Create handler with actual schema
    handler = JSONHandler("schema/output_schema.json")
    
    # Test data with multilingual content and edge cases
    test_title = "PDF Analysis Report: Título con Acentos é Símbolos"
    test_headings = [
        {"level": 1, "text": "Introduction", "page": 1},
        {"level": 2, "text": "Background & Motivation", "page": 2},
        {"level": 3, "text": "Related Work", "page": 3},
        {"level": 1, "text": "Methodology", "page": 5},
        {"level": 2, "text": "Data Collection", "page": 6},
        {"level": 2, "text": "Analysis Techniques", "page": 8},
        {"level": 3, "text": "Statistical Methods", "page": 9},
        {"level": 1, "text": "Results & Discussion", "page": 12},
        {"level": 2, "text": "Key Findings", "page": 13},
        {"level": 1, "text": "Conclusion", "page": 18},
        # Multilingual headings
        {"level": 2, "text": "Résumé des Résultats", "page": 19},
        {"level": 2, "text": "日本語のセクション", "page": 20},
        {"level": 3, "text": "Mathematical: ∑(xi) = ∫f(x)dx", "page": 21}
    ]
    
    print("Testing JSON handler integration...")
    
    # Test 1: Basic formatting and validation
    print("1. Testing basic formatting...")
    json_output = handler.create_json_output(test_title, test_headings, validate=True)
    
    assert json_output["title"] == test_title
    assert len(json_output["outline"]) == len(test_headings)
    
    # Verify level formatting
    levels = [h["level"] for h in json_output["outline"]]
    assert all(level in ["H1", "H2", "H3"] for level in levels)
    
    print("✓ Basic formatting passed")
    
    # Test 2: Multilingual content preservation
    print("2. Testing multilingual content...")
    
    # Find multilingual headings
    french_heading = next(h for h in json_output["outline"] if "Résumé" in h["text"])
    japanese_heading = next(h for h in json_output["outline"] if "日本語" in h["text"])
    math_heading = next(h for h in json_output["outline"] if "∑" in h["text"])
    
    assert "Résumé des Résultats" == french_heading["text"]
    assert "日本語のセクション" == japanese_heading["text"]
    assert "Mathematical: ∑(xi) = ∫f(x)dx" == math_heading["text"]
    
    print("✓ Multilingual content preserved")
    
    # Test 3: Schema validation
    print("3. Testing schema validation...")
    
    validation_result = handler.validate_schema(json_output)
    assert validation_result, "Schema validation should pass"
    
    print("✓ Schema validation passed")
    
    # Test 4: File writing and reading
    print("4. Testing file I/O...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_output.json"
        
        success = handler.write_json_file(json_output, str(output_path))
        assert success, "File writing should succeed"
        
        # Read back and verify
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == json_output
        
        # Verify multilingual content is preserved in file
        assert "Résumé des Résultats" in str(loaded_data)
        assert "日本語のセクション" in str(loaded_data)
        assert "∑(xi) = ∫f(x)dx" in str(loaded_data)
    
    print("✓ File I/O passed")
    
    # Test 5: Edge cases
    print("5. Testing edge cases...")
    
    # Empty title and outline
    empty_output = handler.create_json_output("", [], validate=True)
    assert empty_output["title"] == ""
    assert empty_output["outline"] == []
    
    # Invalid headings mixed with valid ones
    mixed_headings = [
        {"level": 1, "text": "Valid Heading", "page": 1},
        {"level": "invalid", "text": "", "page": 2},  # Invalid
        {"text": "Missing Level", "page": 3},  # Missing level
        {"level": 2, "text": "Another Valid", "page": 4}
    ]
    
    mixed_output = handler.create_json_output("Test", mixed_headings, validate=True)
    assert len(mixed_output["outline"]) == 2  # Only valid headings
    
    print("✓ Edge cases handled")
    
    # Test 6: DocumentStructure integration
    print("6. Testing DocumentStructure integration...")
    
    doc_structure = DocumentStructure(
        title=test_title,
        headings=test_headings,
        metadata={"pages": 25, "processing_time": 3.5}
    )
    
    structure_output = handler.format_from_document_structure(doc_structure)
    assert structure_output["title"] == test_title
    assert len(structure_output["outline"]) == len(test_headings)
    
    print("✓ DocumentStructure integration passed")
    
    print("\n🎉 All integration tests passed!")
    print(f"✓ Processed {len(test_headings)} headings")
    multilingual_count = len([h for h in test_headings if any(ord(c) > 127 for c in h['text'])])
    print(f"✓ Preserved multilingual content in {multilingual_count} headings")
    print(f"✓ Schema validation successful")
    print(f"✓ File I/O with UTF-8 encoding successful")


if __name__ == "__main__":
    test_json_handler_integration()