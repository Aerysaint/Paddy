"""
Tests for title extraction functionality.

This module tests the various title extraction strategies including
metadata extraction, visual prominence analysis, pattern matching,
and filename fallbacks.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.title_extractor import TitleExtractor, extract_title_from_pdf
from src.data_models import TextBlock


class TestTitleExtractor:
    """Test cases for TitleExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TitleExtractor()
        
        # Create sample text blocks for testing
        self.sample_text_blocks = [
            TextBlock(
                text="Document Title Here",
                page=1,
                font_size=18.0,
                font_name="Arial-Bold",
                is_bold=True,
                bbox=(100, 700, 400, 720),
                line_height=22.0
            ),
            TextBlock(
                text="This is regular body text that follows the title.",
                page=1,
                font_size=12.0,
                font_name="Arial",
                is_bold=False,
                bbox=(100, 650, 500, 670),
                line_height=14.0
            ),
            TextBlock(
                text="1. Introduction",
                page=1,
                font_size=14.0,
                font_name="Arial-Bold",
                is_bold=True,
                bbox=(100, 600, 200, 620),
                line_height=16.0
            )
        ]
        
        self.sample_metadata = {
            'title': 'Sample Document Title',
            'author': 'Test Author',
            'subject': 'Test Subject'
        }
    
    def test_extract_from_metadata_success(self):
        """Test successful title extraction from PDF metadata."""
        title = self.extractor._extract_from_metadata(self.sample_metadata)
        assert title == "Sample Document Title"
    
    def test_extract_from_metadata_empty(self):
        """Test metadata extraction with no metadata."""
        title = self.extractor._extract_from_metadata(None)
        assert title == ""
        
        title = self.extractor._extract_from_metadata({})
        assert title == ""
    
    def test_extract_from_metadata_invalid_title(self):
        """Test metadata extraction with invalid title."""
        invalid_metadata = {'title': '123'}  # Too short
        title = self.extractor._extract_from_metadata(invalid_metadata)
        assert title == ""
    
    def test_extract_by_visual_prominence(self):
        """Test title extraction by visual prominence."""
        title = self.extractor._extract_by_visual_prominence(self.sample_text_blocks)
        assert title == "Document Title Here"
    
    def test_extract_by_visual_prominence_empty_blocks(self):
        """Test visual prominence extraction with no blocks."""
        title = self.extractor._extract_by_visual_prominence([])
        assert title == ""
    
    def test_calculate_visual_prominence_score(self):
        """Test visual prominence scoring calculation."""
        title_block = self.sample_text_blocks[0]  # "Document Title Here"
        score = self.extractor._calculate_visual_prominence_score(title_block, self.sample_text_blocks)
        
        # Should have high score due to large font, bold, and position
        assert score > 0.5
    
    def test_extract_by_pattern_matching(self):
        """Test pattern matching extraction."""
        # Create blocks with title-like patterns
        pattern_blocks = [
            TextBlock(
                text="Chapter 1: Introduction to Testing",
                page=1,
                font_size=16.0,
                font_name="Arial-Bold",
                is_bold=True,
                bbox=(100, 700, 400, 720),
                line_height=18.0
            )
        ]
        
        title = self.extractor._extract_by_pattern_matching(pattern_blocks)
        assert "Chapter 1: Introduction to Testing" in title
    
    def test_extract_from_filename(self):
        """Test filename-based title extraction."""
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Test with descriptive filename
            test_path = tmp_path.replace('.pdf', '_research_paper_analysis.pdf')
            title = self.extractor._extract_from_filename(test_path)
            assert "Research Paper Analysis" in title
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_clean_filename_for_title(self):
        """Test filename cleaning for title generation."""
        # Test various filename patterns
        assert "Research Paper" in self.extractor._clean_filename_for_title("research_paper")
        assert "Document Analysis" in self.extractor._clean_filename_for_title("document-analysis")
        assert "Final Report" in self.extractor._clean_filename_for_title("final.report.v2")
    
    def test_is_potential_title_text(self):
        """Test title text validation."""
        assert self.extractor._is_potential_title_text("Valid Title Text")
        assert not self.extractor._is_potential_title_text("123")  # Just numbers
        assert not self.extractor._is_potential_title_text("page 1")  # Page number
        assert not self.extractor._is_potential_title_text("")  # Empty
        assert not self.extractor._is_potential_title_text("a")  # Too short
    
    def test_is_valid_title(self):
        """Test title validation."""
        assert self.extractor._is_valid_title("Valid Document Title")
        assert not self.extractor._is_valid_title("")
        assert not self.extractor._is_valid_title("a")  # Too short
        assert not self.extractor._is_valid_title("x" * 300)  # Too long
    
    def test_clean_title(self):
        """Test title cleaning and normalization."""
        # Test basic cleaning
        assert self.extractor._clean_title("  Title with spaces  ") == "Title with spaces"
        
        # Test prefix removal
        assert self.extractor._clean_title("Title: Document Name") == "Title: Document Name"
        
        # Test punctuation removal
        assert self.extractor._clean_title("Document Title...") == "Document Title"
    
    def test_extract_title_priority_order(self):
        """Test that title extraction follows correct priority order."""
        pdf_path = "test_document.pdf"
        
        # Test metadata priority (should return metadata title)
        title = self.extractor.extract_title(pdf_path, self.sample_text_blocks, self.sample_metadata)
        assert title == "Sample Document Title"
        
        # Test visual prominence when no metadata
        title = self.extractor.extract_title(pdf_path, self.sample_text_blocks, None)
        assert title == "Document Title Here"
    
    def test_get_title_extraction_info(self):
        """Test detailed title extraction information."""
        pdf_path = "test_document.pdf"
        info = self.extractor.get_title_extraction_info(pdf_path, self.sample_text_blocks, self.sample_metadata)
        
        assert 'extracted_title' in info
        assert 'extraction_method' in info
        assert 'candidates' in info
        assert 'metadata_title' in info
        assert 'filename_title' in info
        
        assert info['extracted_title'] == "Sample Document Title"
        assert info['extraction_method'] == "metadata"
    
    def test_multilingual_title_support(self):
        """Test support for multilingual titles."""
        multilingual_blocks = [
            TextBlock(
                text="日本語のタイトル",  # Japanese title
                page=1,
                font_size=18.0,
                font_name="Arial-Bold",
                is_bold=True,
                bbox=(100, 700, 400, 720),
                line_height=22.0
            )
        ]
        
        title = self.extractor._extract_by_visual_prominence(multilingual_blocks)
        assert title == "日本語のタイトル"
    
    def test_special_characters_preservation(self):
        """Test preservation of special characters in titles."""
        special_char_blocks = [
            TextBlock(
                text="Mathematical Analysis: ∑ and ∫ Functions",
                page=1,
                font_size=18.0,
                font_name="Arial-Bold",
                is_bold=True,
                bbox=(100, 700, 400, 720),
                line_height=22.0
            )
        ]
        
        title = self.extractor._extract_by_visual_prominence(special_char_blocks)
        assert "∑" in title and "∫" in title


class TestTitleExtractionConvenienceFunctions:
    """Test convenience functions for title extraction."""
    
    def test_extract_title_from_pdf(self):
        """Test the convenience function for title extraction."""
        sample_blocks = [
            TextBlock(
                text="Test Document Title",
                page=1,
                font_size=18.0,
                font_name="Arial-Bold",
                is_bold=True,
                bbox=(100, 700, 400, 720),
                line_height=22.0
            )
        ]
        
        title = extract_title_from_pdf("test.pdf", sample_blocks)
        assert title == "Test Document Title"