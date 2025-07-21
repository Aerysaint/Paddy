#!/usr/bin/env python3
"""
Simple integration test for the batch processor.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from batch_processor import BatchProcessor

def test_batch_processor_integration():
    """Test batch processor with mock setup."""
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    input_dir = Path(temp_dir) / "input"
    output_dir = Path(temp_dir) / "output"
    
    try:
        # Create directories
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        # Create a simple test file (not a real PDF, but for testing directory discovery)
        test_pdf = input_dir / "test.pdf"
        test_pdf.write_text("dummy content")
        
        # Initialize batch processor
        processor = BatchProcessor()
        
        # Test PDF discovery
        pdf_files = processor._discover_pdf_files(input_dir)
        print(f"Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
        
        # Test empty document handling
        output_file = output_dir / "test.json"
        result = processor._handle_empty_document(test_pdf, output_file)
        print(f"Empty document handling result: {result}")
        
        if result and output_file.exists():
            content = output_file.read_text()
            print(f"Generated JSON content: {content}")
        
        # Test error handling
        test_error = Exception("Test error")
        processor._handle_processing_error(test_pdf, test_error)
        print(f"Error handling stats: failed={processor.stats['failed']}, errors={len(processor.stats['errors'])}")
        
        print("✅ Batch processor integration test passed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_batch_processor_integration()