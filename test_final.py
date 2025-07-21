#!/usr/bin/env python3

from src.batch_processor import BatchProcessor
import json

def test_all_files():
    """Test all files with the improved system"""
    processor = BatchProcessor()
    
    # Test all files
    files = ['file01.pdf', 'file02.pdf', 'file03.pdf', 'file04.pdf', 'file05.pdf']
    for file in files:
        result = processor.process_single_pdf(f'pdfs/{file}', f'final_{file.replace(".pdf", ".json")}')
        print(f'{file}: {"Success" if result else "Failed"}')
    
    # Check file04 result
    with open('final_file04.json', 'r') as f:
        result = json.load(f)
        print(f'\nFile04 result: {result}')

if __name__ == "__main__":
    test_all_files()