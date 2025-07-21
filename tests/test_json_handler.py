"""
Tests for JSON output handler with schema validation.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.json_handler import JSONHandler, create_json_handler
from src.data_models import DocumentStructure, HeadingCandidate


class TestJSONHandler:
    """Test cases for JSONHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = JSONHandler()
        
        # Sample test data
        self.sample_title = "Test Document Title"
        self.sample_headings = [
            {"level": 1, "text": "Introduction", "page": 1},
            {"level": 2, "text": "Background", "page": 2},
            {"level": 3, "text": "Related Work", "page": 3},
            {"level": 1, "text": "Methodology", "page": 4}
        ]
        
        # Multilingual test data
        self.multilingual_title = "Título con Acentos é Símbolos ñ"
        self.multilingual_headings = [
            {"level": 1, "text": "Introducción", "page": 1},
            {"level": 2, "text": "Métodos y Técnicas", "page": 2},
            {"level": 3, "text": "Análisis Estadístico", "page": 3},
            {"level": 1, "text": "日本語のセクション", "page": 4},  # Japanese text
            {"level": 2, "text": "Mathematical: ∑(xi) = ∫f(x)dx", "page": 5}  # Math symbols
        ]
    
    def test_initialization_default(self):
        """Test JSONHandler initialization with default schema path."""
        handler = JSONHandler()
        assert handler.schema_path == "schema/output_schema.json"
    
    def test_initialization_custom_schema(self):
        """Test JSONHandler initialization with custom schema path."""
        custom_path = "custom/schema.json"
        handler = JSONHandler(custom_path)
        assert handler.schema_path == custom_path
    
    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        # Test whitespace normalization
        assert self.handler.normalize_text("  hello   world  ") == "hello world"
        
        # Test empty string
        assert self.handler.normalize_text("") == ""
        assert self.handler.normalize_text(None) == ""
        
        # Test multiple whitespace types
        assert self.handler.normalize_text("hello\t\n  world") == "hello world"
    
    def test_normalize_text_unicode(self):
        """Test Unicode normalization."""
        # Test NFC normalization (composed vs decomposed characters)
        composed = "café"  # é as single character
        decomposed = "cafe\u0301"  # e + combining acute accent
        
        result1 = self.handler.normalize_text(composed)
        result2 = self.handler.normalize_text(decomposed)
        assert result1 == result2 == "café"
    
    def test_sanitize_for_json_basic(self):
        """Test basic JSON sanitization."""
        # Test normal text
        assert self.handler.sanitize_for_json("Hello World") == "Hello World"
        
        # Test empty input
        assert self.handler.sanitize_for_json("") == ""
        assert self.handler.sanitize_for_json(None) == ""
    
    def test_sanitize_for_json_special_characters(self):
        """Test JSON sanitization with special characters."""
        # Test accented characters (should be preserved)
        text = "Café, naïve, résumé"
        result = self.handler.sanitize_for_json(text)
        assert result == "Café, naïve, résumé"
        
        # Test mathematical symbols (should be preserved)
        text = "∑(xi) = ∫f(x)dx"
        result = self.handler.sanitize_for_json(text)
        assert result == "∑(xi) = ∫f(x)dx"
        
        # Test Japanese characters (should be preserved)
        text = "日本語のテスト"
        result = self.handler.sanitize_for_json(text)
        assert result == "日本語のテスト"
    
    def test_sanitize_for_json_control_characters(self):
        """Test JSON sanitization removes problematic control characters."""
        # Test with control characters that should be removed
        text = "Hello\x00World\x01Test"
        result = self.handler.sanitize_for_json(text)
        assert result == "HelloWorldTest"
        
        # Test that valid whitespace is preserved
        text = "Hello\tWorld\nTest"
        result = self.handler.sanitize_for_json(text)
        assert "Hello" in result and "World" in result and "Test" in result
    
    def test_format_heading_level_integer(self):
        """Test heading level formatting from integers."""
        assert self.handler.format_heading_level(1) == "H1"
        assert self.handler.format_heading_level(2) == "H2"
        assert self.handler.format_heading_level(3) == "H3"
        
        # Test edge cases
        assert self.handler.format_heading_level(0) == "H1"  # Below range
        assert self.handler.format_heading_level(4) == "H3"  # Above range
        assert self.handler.format_heading_level(10) == "H3"  # Way above range
    
    def test_format_heading_level_string(self):
        """Test heading level formatting from strings."""
        assert self.handler.format_heading_level("H1") == "H1"
        assert self.handler.format_heading_level("h2") == "H2"
        assert self.handler.format_heading_level("H3") == "H3"
        
        # Test numeric strings
        assert self.handler.format_heading_level("1") == "H1"
        assert self.handler.format_heading_level("2") == "H2"
        assert self.handler.format_heading_level("3") == "H3"
        
        # Test invalid strings
        assert self.handler.format_heading_level("invalid") == "H1"  # Default fallback
    
    def test_format_output_basic(self):
        """Test basic output formatting."""
        result = self.handler.format_output(self.sample_title, self.sample_headings)
        
        assert result["title"] == self.sample_title
        assert len(result["outline"]) == 4
        
        # Check first heading
        first_heading = result["outline"][0]
        assert first_heading["level"] == "H1"
        assert first_heading["text"] == "Introduction"
        assert first_heading["page"] == 1
    
    def test_format_output_multilingual(self):
        """Test output formatting with multilingual content."""
        result = self.handler.format_output(self.multilingual_title, self.multilingual_headings)
        
        assert result["title"] == self.multilingual_title
        assert len(result["outline"]) == 5
        
        # Check Japanese heading is preserved
        japanese_heading = next(h for h in result["outline"] if "日本語" in h["text"])
        assert japanese_heading["text"] == "日本語のセクション"
        
        # Check mathematical symbols are preserved
        math_heading = next(h for h in result["outline"] if "∑" in h["text"])
        assert "∑(xi) = ∫f(x)dx" in math_heading["text"]
    
    def test_format_output_empty_title(self):
        """Test output formatting with empty title."""
        result = self.handler.format_output("", self.sample_headings)
        assert result["title"] == ""
        assert len(result["outline"]) == 4
    
    def test_format_output_empty_headings(self):
        """Test output formatting with empty headings."""
        result = self.handler.format_output(self.sample_title, [])
        assert result["title"] == self.sample_title
        assert result["outline"] == []
    
    def test_format_output_invalid_headings(self):
        """Test output formatting with invalid heading data."""
        invalid_headings = [
            {"level": 1, "text": "Valid Heading", "page": 1},
            {"level": "invalid", "text": "", "page": 2},  # Empty text
            {"text": "Missing Level", "page": 3},  # Missing level
            "not a dict",  # Invalid format
            {"level": 2, "text": "Another Valid", "page": 4}
        ]
        
        result = self.handler.format_output(self.sample_title, invalid_headings)
        
        # Should only include valid headings
        assert len(result["outline"]) == 2
        assert result["outline"][0]["text"] == "Valid Heading"
        assert result["outline"][1]["text"] == "Another Valid"
    
    def test_format_from_document_structure(self):
        """Test formatting from DocumentStructure object."""
        doc_structure = DocumentStructure(
            title=self.sample_title,
            headings=self.sample_headings,
            metadata={"pages": 10}
        )
        
        result = self.handler.format_from_document_structure(doc_structure)
        
        assert result["title"] == self.sample_title
        assert len(result["outline"]) == 4
        assert result["outline"][0]["level"] == "H1"
    
    def test_format_from_candidates(self):
        """Test formatting from HeadingCandidate objects."""
        candidates = [
            HeadingCandidate(
                text="Introduction",
                page=1,
                confidence_score=0.9,
                formatting_features={},
                level_indicators=[],
                assigned_level=1
            ),
            HeadingCandidate(
                text="Background",
                page=2,
                confidence_score=0.8,
                formatting_features={},
                level_indicators=[],
                assigned_level=2
            ),
            HeadingCandidate(
                text="Unassigned",
                page=3,
                confidence_score=0.5,
                formatting_features={},
                level_indicators=[],
                assigned_level=None  # Should be skipped
            )
        ]
        
        result = self.handler.format_from_candidates(self.sample_title, candidates)
        
        assert result["title"] == self.sample_title
        assert len(result["outline"]) == 2  # Only assigned candidates
        assert result["outline"][0]["text"] == "Introduction"
        assert result["outline"][1]["text"] == "Background"
    
    def test_handle_edge_cases_missing_title(self):
        """Test edge case handling for missing title."""
        json_data = {"outline": self.sample_headings}
        result = self.handler.handle_edge_cases(json_data)
        assert result["title"] == ""
    
    def test_handle_edge_cases_missing_outline(self):
        """Test edge case handling for missing outline."""
        json_data = {"title": self.sample_title}
        result = self.handler.handle_edge_cases(json_data)
        assert result["outline"] == []
    
    def test_handle_edge_cases_invalid_outline(self):
        """Test edge case handling for invalid outline format."""
        json_data = {
            "title": self.sample_title,
            "outline": "not a list"
        }
        result = self.handler.handle_edge_cases(json_data)
        assert result["outline"] == []
    
    def test_handle_edge_cases_invalid_outline_entries(self):
        """Test edge case handling for invalid outline entries."""
        json_data = {
            "title": self.sample_title,
            "outline": [
                {"level": 1, "text": "Valid", "page": 1},
                {"level": 2, "page": 2},  # Missing text
                "invalid entry",  # Not a dict
                {"level": 3, "text": "", "page": 3},  # Empty text
                {"level": 2, "text": "Another Valid", "page": 4}
            ]
        }
        
        result = self.handler.handle_edge_cases(json_data)
        
        # Should only keep valid entries
        assert len(result["outline"]) == 2
        assert result["outline"][0]["text"] == "Valid"
        assert result["outline"][1]["text"] == "Another Valid"
    
    def test_create_json_output_complete(self):
        """Test complete JSON output creation."""
        result = self.handler.create_json_output(
            self.sample_title, 
            self.sample_headings, 
            validate=False  # Skip validation for this test
        )
        
        assert result["title"] == self.sample_title
        assert len(result["outline"]) == 4
        assert all(isinstance(h["page"], int) for h in result["outline"])
        assert all(h["level"] in ["H1", "H2", "H3"] for h in result["outline"])
    
    def test_write_json_file_success(self):
        """Test successful JSON file writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.json"
            json_data = {"title": "Test", "outline": []}
            
            success = self.handler.write_json_file(json_data, str(output_path))
            
            assert success
            assert output_path.exists()
            
            # Verify content
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == json_data
    
    def test_write_json_file_creates_directory(self):
        """Test that JSON file writing creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "test_output.json"
            json_data = {"title": "Test", "outline": []}
            
            success = self.handler.write_json_file(json_data, str(output_path))
            
            assert success
            assert output_path.exists()
            assert output_path.parent.exists()
    
    def test_write_json_file_multilingual(self):
        """Test JSON file writing with multilingual content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "multilingual.json"
            json_data = {
                "title": "Título con Acentos",
                "outline": [
                    {"level": "H1", "text": "日本語のセクション", "page": 1},
                    {"level": "H2", "text": "Mathematical: ∑(xi)", "page": 2}
                ]
            }
            
            success = self.handler.write_json_file(json_data, str(output_path))
            
            assert success
            
            # Verify multilingual content is preserved
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            assert loaded_data["title"] == "Título con Acentos"
            assert "日本語のセクション" in loaded_data["outline"][0]["text"]
            assert "∑(xi)" in loaded_data["outline"][1]["text"]
    
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_write_json_file_failure(self, mock_file):
        """Test JSON file writing failure handling."""
        json_data = {"title": "Test", "outline": []}
        success = self.handler.write_json_file(json_data, "/invalid/path/test.json")
        assert not success
    
    def test_process_and_write_success(self):
        """Test complete processing and writing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "complete_test.json"
            
            success = self.handler.process_and_write(
                self.sample_title,
                self.sample_headings,
                str(output_path),
                validate=False
            )
            
            assert success
            assert output_path.exists()
            
            # Verify the complete output
            with open(output_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            assert result["title"] == self.sample_title
            assert len(result["outline"]) == 4
    
    def test_process_and_write_with_edge_cases(self):
        """Test processing and writing with edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "edge_cases.json"
            
            # Test with empty title and mixed valid/invalid headings
            headings = [
                {"level": 1, "text": "Valid Heading", "page": 1},
                {"level": "invalid", "text": "", "page": 2},  # Invalid
                {"level": 2, "text": "Another Valid", "page": 3}
            ]
            
            success = self.handler.process_and_write(
                "",  # Empty title
                headings,
                str(output_path),
                validate=False
            )
            
            assert success
            
            # Verify edge cases are handled
            with open(output_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            assert result["title"] == ""
            assert len(result["outline"]) == 2  # Only valid headings


class TestJSONHandlerSchemaValidation:
    """Test cases for schema validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures with mock schema."""
        # Create a simple mock schema for testing
        self.mock_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "outline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "level": {"type": "string"},
                            "text": {"type": "string"},
                            "page": {"type": "integer"}
                        },
                        "required": ["level", "text", "page"]
                    }
                }
            },
            "required": ["title", "outline"]
        }
    
    @patch('builtins.open', mock_open(read_data='{"type": "object"}'))
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_schema_success(self, mock_exists):
        """Test successful schema loading."""
        handler = JSONHandler("test_schema.json")
        assert handler.schema is not None
    
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_schema_file_not_found(self, mock_exists):
        """Test schema loading when file doesn't exist."""
        handler = JSONHandler("nonexistent.json")
        assert handler.schema is None
    
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_schema_read_error(self, mock_exists, mock_file):
        """Test schema loading with read error."""
        handler = JSONHandler("test_schema.json")
        assert handler.schema is None
    
    def test_validate_schema_success(self):
        """Test successful schema validation."""
        handler = JSONHandler()
        handler.schema = self.mock_schema
        
        valid_data = {
            "title": "Test Document",
            "outline": [
                {"level": "H1", "text": "Introduction", "page": 1}
            ]
        }
        
        assert handler.validate_schema(valid_data)
    
    def test_validate_schema_failure(self):
        """Test schema validation failure."""
        handler = JSONHandler()
        handler.schema = self.mock_schema
        
        invalid_data = {
            "title": "Test Document",
            "outline": [
                {"level": "H1", "text": "Introduction"}  # Missing page
            ]
        }
        
        assert not handler.validate_schema(invalid_data)
    
    def test_validate_schema_no_schema_loaded(self):
        """Test validation when no schema is loaded."""
        handler = JSONHandler()
        handler.schema = None
        
        data = {"title": "Test", "outline": []}
        assert handler.validate_schema(data)  # Should return True when no schema


class TestFactoryFunction:
    """Test cases for factory function."""
    
    def test_create_json_handler_default(self):
        """Test factory function with default parameters."""
        handler = create_json_handler()
        assert isinstance(handler, JSONHandler)
        assert handler.schema_path == "schema/output_schema.json"
    
    def test_create_json_handler_custom_schema(self):
        """Test factory function with custom schema path."""
        custom_path = "custom/schema.json"
        handler = create_json_handler(custom_path)
        assert isinstance(handler, JSONHandler)
        assert handler.schema_path == custom_path


if __name__ == "__main__":
    pytest.main([__file__])