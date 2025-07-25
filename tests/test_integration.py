"""
Integration tests for the PDF layout extraction pipeline.

These tests verify end-to-end functionality by processing sample PDF files
and comparing the output with expected JSON results.
"""

import pytest
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import process_single_pdf
from parser import extract_blocks
from cluster import cluster_font_sizes, assign_size_levels
from detect import filter_headers_footers, detect_title, detect_headings
from outline import build_hierarchy, format_json_structure


class TestEndToEndProcessing:
    """Test complete end-to-end PDF processing pipeline."""
    
    @pytest.fixture
    def sample_files(self):
        """Provide paths to sample PDF files and expected outputs."""
        base_dir = Path(__file__).parent.parent
        pdf_dir = base_dir / "pdfs"
        output_dir = base_dir / "outputs"
        
        return {
            "file01": {
                "pdf": pdf_dir / "file01.pdf",
                "expected": output_dir / "file01.json"
            },
            "file02": {
                "pdf": pdf_dir / "file02.pdf", 
                "expected": output_dir / "file02.json"
            },
            "file03": {
                "pdf": pdf_dir / "file03.pdf",
                "expected": output_dir / "file03.json"
            },
            "file04": {
                "pdf": pdf_dir / "file04.pdf",
                "expected": output_dir / "file04.json"
            },
            "file05": {
                "pdf": pdf_dir / "file05.pdf",
                "expected": output_dir / "file05.json"
            }
        }
    
    def load_expected_output(self, expected_path: Path) -> Dict[str, Any]:
        """Load expected JSON output from file."""
        with open(expected_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_json_outputs(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare actual and expected JSON outputs and return comparison results.
        
        Returns:
            Dictionary with comparison metrics and details
        """
        results = {
            "title_match": actual.get("title", "") == expected.get("title", ""),
            "outline_count_match": len(actual.get("outline", [])) == len(expected.get("outline", [])),
            "exact_matches": 0,
            "total_entries": len(expected.get("outline", [])),
            "accuracy": 0.0,
            "mismatches": []
        }
        
        actual_outline = actual.get("outline", [])
        expected_outline = expected.get("outline", [])
        
        # Compare each outline entry
        for i, expected_entry in enumerate(expected_outline):
            if i < len(actual_outline):
                actual_entry = actual_outline[i]
                
                # Check if entries match exactly
                if (actual_entry.get("level") == expected_entry.get("level") and
                    actual_entry.get("text", "").strip() == expected_entry.get("text", "").strip() and
                    actual_entry.get("page") == expected_entry.get("page")):
                    results["exact_matches"] += 1
                else:
                    results["mismatches"].append({
                        "index": i,
                        "expected": expected_entry,
                        "actual": actual_entry
                    })
            else:
                results["mismatches"].append({
                    "index": i,
                    "expected": expected_entry,
                    "actual": None
                })
        
        # Add any extra entries in actual output
        for i in range(len(expected_outline), len(actual_outline)):
            results["mismatches"].append({
                "index": i,
                "expected": None,
                "actual": actual_outline[i]
            })
        
        # Calculate accuracy
        if results["total_entries"] > 0:
            results["accuracy"] = results["exact_matches"] / results["total_entries"]
        else:
            results["accuracy"] = 1.0 if len(actual_outline) == 0 else 0.0
        
        return results
    
    @pytest.mark.parametrize("file_key", ["file01", "file02", "file03", "file04", "file05"])
    def test_end_to_end_processing(self, sample_files, file_key, tmp_path):
        """Test end-to-end processing of sample PDF files."""
        file_info = sample_files[file_key]
        pdf_path = file_info["pdf"]
        expected_path = file_info["expected"]
        
        # Skip test if files don't exist
        if not pdf_path.exists():
            pytest.skip(f"PDF file not found: {pdf_path}")
        if not expected_path.exists():
            pytest.skip(f"Expected output file not found: {expected_path}")
        
        # Load expected output
        expected_output = self.load_expected_output(expected_path)
        
        # Process PDF using the main pipeline
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        success = process_single_pdf(pdf_path, output_dir)
        assert success, f"Failed to process {pdf_path}"
        
        # Load actual output
        output_file = output_dir / f"{pdf_path.stem}.json"
        assert output_file.exists(), f"Output file not created: {output_file}"
        
        with open(output_file, 'r', encoding='utf-8') as f:
            actual_output = json.load(f)
        
        # Compare outputs
        comparison = self.compare_json_outputs(actual_output, expected_output)
        
        # Assert accuracy requirement (≥95%)
        assert comparison["accuracy"] >= 0.95, (
            f"Accuracy {comparison['accuracy']:.2%} < 95% for {file_key}. "
            f"Exact matches: {comparison['exact_matches']}/{comparison['total_entries']}. "
            f"Mismatches: {comparison['mismatches']}"
        )
        
        # Additional assertions
        assert comparison["title_match"], (
            f"Title mismatch for {file_key}: "
            f"expected '{expected_output.get('title', '')}', "
            f"got '{actual_output.get('title', '')}'"
        )
    
    def test_pipeline_components_integration(self, sample_files):
        """Test that pipeline components work together correctly."""
        # Use file01 as a representative test case
        pdf_path = sample_files["file01"]["pdf"]
        
        if not pdf_path.exists():
            pytest.skip(f"PDF file not found: {pdf_path}")
        
        # Step 1: Extract blocks
        blocks = extract_blocks(str(pdf_path))
        assert len(blocks) > 0, "No blocks extracted from PDF"
        
        # Step 2: Filter headers/footers
        filtered_blocks, blocks_with_flags = filter_headers_footers(blocks)
        assert len(filtered_blocks) <= len(blocks), "Filtering should not increase block count"
        
        # Step 3: Cluster font sizes
        size_to_level = cluster_font_sizes(filtered_blocks)
        blocks_with_levels = assign_size_levels(filtered_blocks, size_to_level)
        assert all('size_level' in block for block in blocks_with_levels), "All blocks should have size_level"
        
        # Step 4: Detect title
        title = detect_title(blocks_with_levels)
        assert isinstance(title, str), "Title should be a string"
        
        # Step 5: Detect headings
        headings = detect_headings(blocks_with_levels)
        assert isinstance(headings, list), "Headings should be a list"
        
        # Step 6: Build hierarchy
        hierarchy = build_hierarchy(headings)
        assert isinstance(hierarchy, list), "Hierarchy should be a list"
        
        # Step 7: Format JSON
        json_output = format_json_structure(title, hierarchy)
        assert "title" in json_output, "JSON output should have title field"
        assert "outline" in json_output, "JSON output should have outline field"
        assert isinstance(json_output["outline"], list), "Outline should be a list"


class TestErrorHandling:
    """Test error handling with various edge cases."""
    
    def test_corrupted_pdf_handling(self, tmp_path):
        """Test handling of corrupted PDF files."""
        # Create a fake corrupted PDF file
        corrupted_pdf = tmp_path / "corrupted.pdf"
        corrupted_pdf.write_text("This is not a valid PDF file")
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Process should handle the error gracefully
        success = process_single_pdf(corrupted_pdf, output_dir)
        
        # Should create empty output for corrupted PDFs
        output_file = output_dir / "corrupted.json"
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            assert output_data == {"title": "", "outline": []}
    
    def test_nonexistent_pdf_handling(self, tmp_path):
        """Test handling of non-existent PDF files."""
        nonexistent_pdf = tmp_path / "nonexistent.pdf"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Should return False for non-existent files
        success = process_single_pdf(nonexistent_pdf, output_dir)
        assert success == False
    
    def test_empty_pdf_handling(self, tmp_path):
        """Test handling of PDFs with no extractable text."""
        # This test would require a valid but empty PDF file
        # For now, we'll test the pipeline with empty blocks
        from outline import format_json_structure
        
        # Test with empty inputs
        json_output = format_json_structure("", [])
        assert json_output == {"title": "", "outline": []}
        
        # Test with empty title but some headings
        headings = [
            {
                'level': 1, 'text': 'Test Heading', 'page': 1,
                'source_block': {'y0': 100}
            }
        ]
        json_output = format_json_structure("", headings)
        assert json_output["title"] == ""
        assert len(json_output["outline"]) == 1


class TestAccuracyMetrics:
    """Test accuracy measurement and reporting."""
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation logic."""
        from tests.test_integration import TestEndToEndProcessing
        
        test_instance = TestEndToEndProcessing()
        
        # Test perfect match
        actual = {
            "title": "Test Document",
            "outline": [
                {"level": "H1", "text": "Chapter 1", "page": 1},
                {"level": "H2", "text": "Section 1.1", "page": 2}
            ]
        }
        expected = actual.copy()
        
        comparison = test_instance.compare_json_outputs(actual, expected)
        assert comparison["accuracy"] == 1.0
        assert comparison["exact_matches"] == 2
        assert len(comparison["mismatches"]) == 0
        
        # Test partial match
        actual_partial = {
            "title": "Test Document",
            "outline": [
                {"level": "H1", "text": "Chapter 1", "page": 1},
                {"level": "H2", "text": "Different Section", "page": 2}  # Mismatch
            ]
        }
        
        comparison = test_instance.compare_json_outputs(actual_partial, expected)
        assert comparison["accuracy"] == 0.5  # 1 out of 2 matches
        assert comparison["exact_matches"] == 1
        assert len(comparison["mismatches"]) == 1
    
    def test_empty_outline_accuracy(self):
        """Test accuracy calculation with empty outlines."""
        from tests.test_integration import TestEndToEndProcessing
        
        test_instance = TestEndToEndProcessing()
        
        # Both empty - should be 100% accurate
        actual = {"title": "", "outline": []}
        expected = {"title": "", "outline": []}
        
        comparison = test_instance.compare_json_outputs(actual, expected)
        assert comparison["accuracy"] == 1.0
        
        # Expected empty, actual has content - should be 0% accurate
        actual_with_content = {"title": "", "outline": [{"level": "H1", "text": "Test", "page": 1}]}
        
        comparison = test_instance.compare_json_outputs(actual_with_content, expected)
        assert comparison["accuracy"] == 0.0  # Expected empty but actual has content = 0% accuracy


if __name__ == '__main__':
    pytest.main([__file__])