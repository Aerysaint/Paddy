"""
Unit tests for PDF extractor functionality.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pdf_extractor import PDFExtractor, extract_pdf_text, get_pdf_title
from data_models import TextBlock


class TestPDFExtractor(unittest.TestCase):
    """Test cases for PDF extractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PDFExtractor()
        self.test_pdf_dir = Path(__file__).parent.parent / 'pdfs'
        
        # Find a test PDF file
        self.test_pdf = None
        if self.test_pdf_dir.exists():
            pdf_files = list(self.test_pdf_dir.glob('*.pdf'))
            if pdf_files:
                self.test_pdf = pdf_files[0]
    
    def test_extractor_initialization(self):
        """Test PDFExtractor initialization."""
        self.assertIsInstance(self.extractor, PDFExtractor)
        self.assertEqual(self.extractor.supported_extensions, {'.pdf'})
        self.assertIsInstance(self.extractor.encoding_fallbacks, list)
    
    def test_extract_text_with_metadata(self):
        """Test text extraction with metadata."""
        if not self.test_pdf:
            self.skipTest("No test PDF files available")
        
        text_blocks = self.extractor.extract_text_with_metadata(str(self.test_pdf))
        
        # Verify we got text blocks
        self.assertIsInstance(text_blocks, list)
        self.assertGreater(len(text_blocks), 0)
        
        # Verify each text block has required attributes
        for block in text_blocks[:5]:  # Check first 5 blocks
            self.assertIsInstance(block, TextBlock)
            self.assertIsInstance(block.text, str)
            self.assertGreater(len(block.text.strip()), 0)
            self.assertIsInstance(block.page, int)
            self.assertGreater(block.page, 0)
            self.assertIsInstance(block.font_size, float)
            self.assertGreater(block.font_size, 0)
            self.assertIsInstance(block.font_name, str)
            self.assertIsInstance(block.is_bold, bool)
            self.assertIsInstance(block.bbox, tuple)
            self.assertEqual(len(block.bbox), 4)
            self.assertIsInstance(block.line_height, float)
            self.assertGreater(block.line_height, 0)
    
    def test_get_document_title(self):
        """Test document title extraction."""
        if not self.test_pdf:
            self.skipTest("No test PDF files available")
        
        title = self.extractor.get_document_title(str(self.test_pdf))
        
        # Title should be a string (may be empty)
        self.assertIsInstance(title, str)
    
    def test_get_document_info(self):
        """Test document info extraction."""
        if not self.test_pdf:
            self.skipTest("No test PDF files available")
        
        info = self.extractor.get_document_info(str(self.test_pdf))
        
        # Verify required fields
        self.assertIsInstance(info, dict)
        self.assertIn('page_count', info)
        self.assertIn('title', info)
        self.assertIn('metadata', info)
        self.assertIn('file_size', info)
        self.assertIn('file_name', info)
        
        # Verify data types
        self.assertIsInstance(info['page_count'], int)
        self.assertGreater(info['page_count'], 0)
        self.assertIsInstance(info['title'], str)
        self.assertIsInstance(info['metadata'], dict)
        self.assertIsInstance(info['file_size'], int)
        self.assertGreater(info['file_size'], 0)
        self.assertIsInstance(info['file_name'], str)
    
    def test_text_normalization(self):
        """Test text normalization functionality."""
        # Test various text normalization scenarios
        test_cases = [
            ("  Hello   World  ", "Hello World"),
            ("Text\u200bwith\u200czero\u200dwidth", "Textwithzerowidth"),
            ("Multiple\n\n\nlines", "Multiple lines"),
            ("", ""),
            ("   ", ""),
        ]
        
        for input_text, expected in test_cases:
            result = self.extractor._normalize_text(input_text)
            self.assertEqual(result, expected)
    
    def test_font_info_extraction(self):
        """Test font information extraction."""
        # Mock span data
        test_span = {
            'font': 'Arial-Bold',
            'size': 12.0,
            'flags': 16  # Bold flag
        }
        
        font_info = self.extractor._extract_font_info(test_span)
        
        self.assertIsInstance(font_info, dict)
        self.assertIn('name', font_info)
        self.assertIn('size', font_info)
        self.assertIn('is_bold', font_info)
        self.assertEqual(font_info['size'], 12.0)
        self.assertTrue(font_info['is_bold'])
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        if not self.test_pdf:
            self.skipTest("No test PDF files available")
        
        # Test extract_pdf_text function
        text_blocks = extract_pdf_text(str(self.test_pdf))
        self.assertIsInstance(text_blocks, list)
        
        # Test get_pdf_title function
        title = get_pdf_title(str(self.test_pdf))
        self.assertIsInstance(title, str)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_text_with_metadata("nonexistent.pdf")
        
        # Test with invalid file type - create a dummy text file first
        dummy_file = Path("test_dummy.txt")
        dummy_file.write_text("dummy content")
        try:
            with self.assertRaises(ValueError):
                self.extractor.extract_text_with_metadata(str(dummy_file))
        finally:
            if dummy_file.exists():
                dummy_file.unlink()


if __name__ == '__main__':
    unittest.main()