#!/usr/bin/env python3
"""
Integration tests for improved PDF layout extraction.

This test compares our current output with expected outputs to identify
specific areas where the heading detection needs improvement.
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import process_single_pdf


def load_json(file_path):
    """Load JSON file and return parsed content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_outputs(expected_file, actual_file, pdf_name):
    """Compare expected vs actual output and report differences."""
    print(f"\n=== Analysis for {pdf_name} ===")
    
    if not os.path.exists(expected_file):
        print(f"Expected file not found: {expected_file}")
        return
    
    if not os.path.exists(actual_file):
        print(f"Actual file not found: {actual_file}")
        return
    
    expected = load_json(expected_file)
    actual = load_json(actual_file)
    
    print(f"Expected title: '{expected.get('title', '')}'")
    print(f"Actual title: '{actual.get('title', '')}'")
    
    expected_outline = expected.get('outline', [])
    actual_outline = actual.get('outline', [])
    
    print(f"Expected headings: {len(expected_outline)}")
    print(f"Actual headings: {len(actual_outline)}")
    
    if len(expected_outline) > 0:
        print("\nExpected headings:")
        for i, heading in enumerate(expected_outline[:10]):  # Show first 10
            print(f"  {i+1}. {heading.get('level', 'H?')}: '{heading.get('text', '')}' (page {heading.get('page', '?')})")
        if len(expected_outline) > 10:
            print(f"  ... and {len(expected_outline) - 10} more")
    
    if len(actual_outline) > 0:
        print("\nActual headings:")
        for i, heading in enumerate(actual_outline):
            print(f"  {i+1}. {heading.get('level', 'H?')}: '{heading.get('text', '')}' (page {heading.get('page', '?')})")
    
    # Calculate match percentage
    if len(expected_outline) > 0:
        match_percentage = (len(actual_outline) / len(expected_outline)) * 100
        print(f"\nMatch percentage: {match_percentage:.1f}% ({len(actual_outline)}/{len(expected_outline)})")
    
    return {
        'pdf_name': pdf_name,
        'expected_count': len(expected_outline),
        'actual_count': len(actual_outline),
        'match_percentage': match_percentage if len(expected_outline) > 0 else 0
    }


def main():
    """Run integration test analysis."""
    print("PDF Layout Extractor - Integration Test Analysis")
    print("=" * 60)
    
    # Test files
    test_files = ['file01', 'file02', 'file03', 'file04', 'file05']
    
    results = []
    
    for file_name in test_files:
        expected_file = f"outputs/{file_name}.json"
        actual_file = f"test_output_fixed_parser/{file_name}.json"
        
        result = compare_outputs(expected_file, actual_file, file_name)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_expected = sum(r['expected_count'] for r in results)
    total_actual = sum(r['actual_count'] for r in results)
    
    print(f"Total expected headings: {total_expected}")
    print(f"Total actual headings: {total_actual}")
    print(f"Overall detection rate: {(total_actual/total_expected)*100:.1f}%")
    
    print("\nPer-file results:")
    for result in results:
        print(f"  {result['pdf_name']}: {result['actual_count']}/{result['expected_count']} ({result['match_percentage']:.1f}%)")


if __name__ == "__main__":
    main()