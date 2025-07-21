"""
Unit tests for the batch processor module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from batch_processor import BatchProcessor, process_pdf_directory
from data_models import TextBlock, HeadingCandidate


class TestBatchProcessor(unittest.TestCase):
    """Test cases for BatchProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Create directories
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create mock PDF files
        self.pdf1 = self.input_dir / "test1.pdf"
        self.pdf2 = self.input_dir / "test2.pdf"
        self.pdf3 = self.input_dir / "invalid.txt"  # Non-PDF file
        
        # Create empty files
        self.pdf1.touch()
        self.pdf2.touch()
        self.pdf3.touch()
        
        self.processor = BatchProcessor()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_discover_pdf_files(self):
        """Test PDF file discovery."""
        pdf_files = self.processor._discover_pdf_files(self.input_dir)
        
        # Should find 2 PDF files, not the .txt file
        self.assertEqual(len(pdf_files), 2)
        
        # Check file names
        file_names = [f.name for f in pdf_files]
        self.assertIn("test1.pdf", file_names)
        self.assertIn("test2.pdf", file_names)
        self.assertNotIn("invalid.txt", file_names)
    
    def test_discover_pdf_files_empty_directory(self):
        """Test PDF file discovery in empty directory."""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        pdf_files = self.processor._discover_pdf_files(empty_dir)
        self.assertEqual(len(pdf_files), 0)
    
    def test_discover_pdf_files_case_insensitive(self):
        """Test PDF file discovery is case insensitive."""
        # Create PDF files with different cases
        (self.input_dir / "test.PDF").touch()
        (self.input_dir / "test.Pdf").touch()
        
        pdf_files = self.processor._discover_pdf_files(self.input_dir)
        
        # Should find all PDF files regardless of case
        self.assertEqual(len(pdf_files), 4)  # 2 original + 2 new
    
    @patch('batch_processor.PDFExtractor')
    @patch('batch_processor.StructureAnalyzer')
    @patch('batch_processor.HeadingLevelClassifier')
    @patch('batch_processor.TitleExtractor')
    @patch('batch_processor.JSONHandler')
    def test_process_single_pdf_success(self, mock_json, mock_title, mock_classifier, mock_analyzer, mock_extractor):
        """Test successful processing of a single PDF."""
        # Mock the components
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        mock_analyzer_instance = Mock()
        mock_analyzer.return_value = mock_analyzer_instance
        
        mock_classifier_instance = Mock()
        mock_classifier.return_value = mock_classifier_instance
        
        mock_title_instance = Mock()
        mock_title.return_value = mock_title_instance
        
        mock_json_instance = Mock()
        mock_json.return_value = mock_json_instance
        
        # Set up mock returns
        mock_text_blocks = [
            TextBlock("Test heading", 1, 14.0, "Arial", True, (0, 0, 100, 20), 16.0)
        ]
        mock_extractor_instance.extract_text_with_metadata.return_value = mock_text_blocks
        mock_extractor_instance.get_document_info.return_value = {"title": "Test Document"}
        
        mock_title_instance.extract_title.return_value = "Test Document"
        
        mock_candidates = [
            HeadingCandidate("Test heading", 1, 0.8, {}, [], mock_text_blocks[0], 1)
        ]
        mock_analyzer_instance.analyze_document_structure.return_value = (mock_candidates, Mock())
        
        mock_classifier_instance.classify_heading_levels.return_value = mock_candidates
        
        mock_json_instance.process_and_write.return_value = True
        
        # Test processing
        result = self.processor._process_single_pdf(self.pdf1, self.output_dir)
        
        self.assertTrue(result)
        
        # Verify all components were called
        mock_extractor_instance.extract_text_with_metadata.assert_called_once()
        mock_title_instance.extract_title.assert_called_once()
        mock_analyzer_instance.analyze_document_structure.assert_called_once()
        mock_classifier_instance.classify_heading_levels.assert_called_once()
        mock_json_instance.process_and_write.assert_called_once()
    
    def test_handle_empty_document(self):
        """Test handling of documents with no extractable text."""
        output_file = self.output_dir / "empty.json"
        
        with patch.object(self.processor.json_handler, 'process_and_write') as mock_write:
            mock_write.return_value = True
            
            result = self.processor._handle_empty_document(self.pdf1, output_file)
            
            self.assertTrue(result)
            mock_write.assert_called_once()
            
            # Check that title was derived from filename
            args, kwargs = mock_write.call_args
            title = args[0]
            headings = args[1]
            
            self.assertEqual(title, "Test1")  # Filename without extension, formatted
            self.assertEqual(headings, [])
    
    def test_handle_processing_error(self):
        """Test error handling for individual file processing."""
        test_error = Exception("Test error")
        
        # Reset stats
        self.processor.stats['failed'] = 0
        self.processor.stats['errors'] = []
        
        self.processor._handle_processing_error(self.pdf1, test_error)
        
        # Check stats were updated
        self.assertEqual(self.processor.stats['failed'], 1)
        self.assertEqual(len(self.processor.stats['errors']), 1)
        
        error_info = self.processor.stats['errors'][0]
        self.assertEqual(error_info['file'], str(self.pdf1))
        self.assertEqual(error_info['error'], "Test error")
        self.assertEqual(error_info['error_type'], "Exception")
    
    def test_process_directory_invalid_input(self):
        """Test processing with invalid input directory."""
        invalid_dir = "/nonexistent/directory"
        
        with self.assertRaises(FileNotFoundError):
            self.processor.process_directory(invalid_dir, str(self.output_dir))
    
    def test_process_directory_file_as_input(self):
        """Test processing with file instead of directory as input."""
        with self.assertRaises(NotADirectoryError):
            self.processor.process_directory(str(self.pdf1), str(self.output_dir))
    
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        # Set some test stats
        self.processor.stats['total_files'] = 5
        self.processor.stats['successful'] = 3
        self.processor.stats['failed'] = 2
        self.processor.stats['start_time'] = 100.0
        self.processor.stats['end_time'] = 110.0
        
        stats = self.processor.get_processing_stats()
        
        self.assertEqual(stats['total_files'], 5)
        self.assertEqual(stats['successful'], 3)
        self.assertEqual(stats['failed'], 2)
        self.assertEqual(stats['total_time'], 10.0)
        self.assertEqual(stats['success_rate'], 60.0)
        self.assertEqual(stats['average_time_per_file'], 2.0)
    
    def test_reset_stats(self):
        """Test resetting processing statistics."""
        # Set some stats
        self.processor.stats['total_files'] = 5
        self.processor.stats['successful'] = 3
        self.processor.stats['errors'] = [{'test': 'error'}]
        
        self.processor.reset_stats()
        
        # Check all stats are reset
        self.assertEqual(self.processor.stats['total_files'], 0)
        self.assertEqual(self.processor.stats['successful'], 0)
        self.assertEqual(self.processor.stats['failed'], 0)
        self.assertEqual(self.processor.stats['errors'], [])
        self.assertIsNone(self.processor.stats['start_time'])
        self.assertIsNone(self.processor.stats['end_time'])


class TestBatchProcessorConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('batch_processor.BatchProcessor')
    def test_process_pdf_directory(self, mock_processor_class):
        """Test the convenience function for directory processing."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.process_directory.return_value = {'total_files': 2, 'successful': 2}
        
        result = process_pdf_directory(str(self.input_dir), str(self.output_dir), max_workers=2)
        
        mock_processor_class.assert_called_once_with(max_workers=2)
        mock_processor.process_directory.assert_called_once_with(str(self.input_dir), str(self.output_dir))
        self.assertEqual(result, {'total_files': 2, 'successful': 2})


if __name__ == '__main__':
    unittest.main()