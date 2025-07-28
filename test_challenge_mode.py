"""
Test script to demonstrate the challenge mode functionality.
"""

import json
import subprocess
import sys
from pathlib import Path


def test_challenge_mode():
    """Test the challenge mode with the provided input."""
    
    print("Testing Challenge Mode")
    print("=" * 50)
    
    # Check if challenge input exists
    input_file = "Collection 1/challenge1b_input.json"
    if not Path(input_file).exists():
        print(f"Error: Challenge input file not found: {input_file}")
        return False
    
    # Load and display input
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    print("Input Data:")
    print(f"  Challenge ID: {input_data['challenge_info']['challenge_id']}")
    print(f"  Test Case: {input_data['challenge_info']['test_case_name']}")
    print(f"  Description: {input_data['challenge_info']['description']}")
    print(f"  Persona: {input_data['persona']['role']}")
    print(f"  Job to be done: {input_data['job_to_be_done']['task']}")
    print(f"  Documents: {len(input_data['documents'])}")
    
    # Run challenge mode
    print("\nRunning Challenge Mode...")
    cmd = [
        sys.executable, "main.py", input_file,
        "--final-results", "5",
        "--output-file", "test_challenge_output.json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Challenge mode failed:")
        print(result.stderr)
        return False
    
    print("✅ Challenge mode completed successfully!")
    
    # Load and analyze output
    output_file = "test_challenge_output.json"
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            output_data = json.load(f)
        
        print("\nOutput Analysis:")
        print(f"  Metadata persona: {output_data['metadata']['persona']}")
        print(f"  Job to be done: {output_data['metadata']['job_to_be_done']}")
        print(f"  Input documents: {len(output_data['metadata']['input_documents'])}")
        print(f"  Extracted sections: {len(output_data['extracted_sections'])}")
        print(f"  Subsection analysis: {len(output_data['subsection_analysis'])}")
        
        print("\nTop 3 Extracted Sections:")
        for i, section in enumerate(output_data['extracted_sections'][:3], 1):
            print(f"  {i}. {section['document']} - {section['section_title'][:50]}...")
        
        # Verify format matches expected structure
        expected_keys = ['metadata', 'extracted_sections', 'subsection_analysis']
        if all(key in output_data for key in expected_keys):
            print("\n✅ Output format matches expected structure")
        else:
            print("\n❌ Output format does not match expected structure")
            return False
        
        # Verify metadata structure
        metadata_keys = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
        if all(key in output_data['metadata'] for key in metadata_keys):
            print("✅ Metadata format is correct")
        else:
            print("❌ Metadata format is incorrect")
            return False
        
        # Verify extracted sections structure
        if output_data['extracted_sections']:
            section_keys = ['document', 'section_title', 'importance_rank', 'page_number']
            first_section = output_data['extracted_sections'][0]
            if all(key in first_section for key in section_keys):
                print("✅ Extracted sections format is correct")
            else:
                print("❌ Extracted sections format is incorrect")
                return False
        
        # Verify subsection analysis structure
        if output_data['subsection_analysis']:
            subsection_keys = ['document', 'refined_text', 'page_number']
            first_subsection = output_data['subsection_analysis'][0]
            if all(key in first_subsection for key in subsection_keys):
                print("✅ Subsection analysis format is correct")
            else:
                print("❌ Subsection analysis format is incorrect")
                return False
        
        return True
    else:
        print(f"❌ Output file not found: {output_file}")
        return False


def compare_with_expected():
    """Compare output with expected format."""
    
    expected_file = "Collection 1/challenge1b_output.json"
    actual_file = "test_challenge_output.json"
    
    if not Path(expected_file).exists() or not Path(actual_file).exists():
        print("Cannot compare - missing files")
        return
    
    with open(expected_file, 'r') as f:
        expected = json.load(f)
    
    with open(actual_file, 'r') as f:
        actual = json.load(f)
    
    print("\nFormat Comparison:")
    print(f"Expected structure keys: {list(expected.keys())}")
    print(f"Actual structure keys: {list(actual.keys())}")
    
    if set(expected.keys()) == set(actual.keys()):
        print("✅ Top-level structure matches")
    else:
        print("❌ Top-level structure differs")
    
    # Compare metadata structure
    if 'metadata' in both:
        exp_meta_keys = set(expected['metadata'].keys())
        act_meta_keys = set(actual['metadata'].keys())
        
        if exp_meta_keys == act_meta_keys:
            print("✅ Metadata structure matches")
        else:
            print("❌ Metadata structure differs")
            print(f"  Expected: {exp_meta_keys}")
            print(f"  Actual: {act_meta_keys}")


if __name__ == "__main__":
    success = test_challenge_mode()
    
    if success:
        print("\n" + "=" * 50)
        print("CHALLENGE MODE TEST PASSED")
        print("=" * 50)
        compare_with_expected()
    else:
        print("\n" + "=" * 50)
        print("CHALLENGE MODE TEST FAILED")
        print("=" * 50)
        sys.exit(1)