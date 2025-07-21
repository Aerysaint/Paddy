"""
JSON output handler with schema validation for PDF Outline Extractor.

This module handles the formatting and validation of extracted PDF structure
data into JSON format, with special attention to multilingual content and
character encoding preservation.
"""

import json
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import jsonschema
from jsonschema import validate, ValidationError

from .data_models import DocumentStructure, HeadingCandidate
from .logging_config import setup_logging

logger = setup_logging()


class JSONHandler:
    """
    Handles JSON output formatting and schema validation for PDF outline extraction.
    
    Features:
    - Strict schema compliance validation
    - Special character encoding preservation
    - Multilingual content support
    - Edge case handling (empty outlines, missing titles)
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the JSON handler.
        
        Args:
            schema_path: Path to the JSON schema file for validation
        """
        self.schema_path = schema_path or "schema/output_schema.json"
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Optional[Dict[str, Any]]:
        """
        Load the JSON schema for validation.
        
        Returns:
            Loaded schema dictionary or None if loading fails
        """
        try:
            schema_file = Path(self.schema_path)
            if not schema_file.exists():
                logger.warning(f"Schema file not found: {self.schema_path}")
                return None
                
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                logger.debug(f"Loaded schema from {self.schema_path}")
                return schema
                
        except Exception as e:
            logger.error(f"Failed to load schema from {self.schema_path}: {e}")
            return None
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent encoding and character representation.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text with preserved special characters
        """
        if not text:
            return ""
            
        # Apply Unicode NFC normalization to handle composed vs decomposed characters
        normalized = unicodedata.normalize('NFC', text)
        
        # Strip leading/trailing whitespace but preserve internal formatting
        normalized = normalized.strip()
        
        # Replace multiple consecutive whitespace with single space
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def sanitize_for_json(self, text: str) -> str:
        """
        Sanitize text for JSON output while preserving special characters.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            JSON-safe text with preserved multilingual content
        """
        if not text:
            return ""
            
        # Normalize the text first
        sanitized = self.normalize_text(text)
        
        # Handle control characters that might break JSON
        # Remove or replace problematic control characters but preserve printable Unicode
        control_chars = []
        for char in sanitized:
            if unicodedata.category(char).startswith('C') and char not in ['\t', '\n', '\r']:
                control_chars.append(char)
        
        for char in control_chars:
            sanitized = sanitized.replace(char, '')
        
        # Ensure the text is valid for JSON encoding
        try:
            json.dumps(sanitized, ensure_ascii=False)
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            logger.warning(f"Text encoding issue, applying fallback sanitization: {e}")
            # Fallback: encode to UTF-8 and decode, replacing errors
            sanitized = sanitized.encode('utf-8', errors='replace').decode('utf-8')
        
        return sanitized
    
    def format_heading_level(self, level: Union[int, str]) -> str:
        """
        Format heading level to match schema requirements.
        
        Args:
            level: Heading level (1, 2, 3 or "H1", "H2", "H3")
            
        Returns:
            Formatted level string ("H1", "H2", or "H3")
        """
        if isinstance(level, str):
            level = level.upper()
            if level in ["H1", "H2", "H3"]:
                return level
            # Try to extract number from string like "1", "2", "3"
            try:
                level = int(level.replace('H', ''))
            except (ValueError, AttributeError):
                level = 1  # Default fallback
        
        # Convert integer level to string format
        level = int(level)
        if level == 1:
            return "H1"
        elif level == 2:
            return "H2"
        elif level == 3:
            return "H3"
        else:
            # Clamp to valid range
            if level < 1:
                return "H1"
            else:
                return "H3"
    
    def format_output(self, title: str, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format the extracted data into the required JSON structure.
        
        Args:
            title: Document title
            headings: List of heading dictionaries with level, text, and page
            
        Returns:
            Formatted dictionary matching the output schema
        """
        # Handle missing or empty title
        formatted_title = self.sanitize_for_json(title) if title else ""
        
        # Format headings
        formatted_outline = []
        for heading in headings:
            if not isinstance(heading, dict):
                logger.warning(f"Invalid heading format: {heading}")
                continue
                
            # Check for required fields
            if 'level' not in heading or 'text' not in heading or 'page' not in heading:
                logger.debug(f"Skipping heading with missing required fields: {heading}")
                continue
                
            # Extract required fields with validation
            try:
                level = self.format_heading_level(heading['level'])
                text = self.sanitize_for_json(heading['text'])
                page = int(heading['page'])
                
                # Skip empty headings
                if not text:
                    logger.debug("Skipping empty heading")
                    continue
                
                # Ensure page is non-negative (allow 0-based page numbering)
                page = max(0, page)
                
                formatted_outline.append({
                    "level": level,
                    "text": text,
                    "page": page
                })
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error formatting heading {heading}: {e}")
                continue
        
        # Create the final output structure
        output = {
            "title": formatted_title,
            "outline": formatted_outline
        }
        
        return output
    
    def format_from_document_structure(self, doc_structure: DocumentStructure) -> Dict[str, Any]:
        """
        Format output from a DocumentStructure object.
        
        Args:
            doc_structure: DocumentStructure containing title and headings
            
        Returns:
            Formatted dictionary matching the output schema
        """
        return self.format_output(doc_structure.title, doc_structure.headings)
    
    def format_from_candidates(self, title: str, candidates: List[HeadingCandidate]) -> Dict[str, Any]:
        """
        Format output from a list of HeadingCandidate objects.
        
        Args:
            title: Document title
            candidates: List of HeadingCandidate objects
            
        Returns:
            Formatted dictionary matching the output schema
        """
        headings = []
        for candidate in candidates:
            logger.info(f"Processing candidate: '{candidate.text}' - assigned_level: {candidate.assigned_level}")
            if candidate.assigned_level:
                headings.append({
                    'level': candidate.assigned_level,
                    'text': candidate.text,
                    'page': candidate.page
                })
                logger.info(f"Added to headings: H{candidate.assigned_level} - '{candidate.text}'")
        
        return self.format_output(title, headings)
    
    def validate_schema(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate JSON data against the loaded schema.
        
        Args:
            json_data: Dictionary to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        if not self.schema:
            logger.warning("No schema loaded, skipping validation")
            return True
        
        try:
            validate(instance=json_data, schema=self.schema)
            logger.debug("Schema validation passed")
            return True
            
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e.message}")
            logger.debug(f"Validation error path: {e.absolute_path}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error during schema validation: {e}")
            return False
    
    def handle_edge_cases(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle edge cases in the JSON data.
        
        Args:
            json_data: Input JSON data
            
        Returns:
            JSON data with edge cases handled
        """
        # Handle missing title
        if not json_data.get('title'):
            json_data['title'] = ""
            logger.debug("Set empty title for missing title")
        
        # Handle missing or empty outline
        if not json_data.get('outline'):
            json_data['outline'] = []
            logger.debug("Set empty outline for missing outline")
        
        # Ensure outline is a list
        if not isinstance(json_data['outline'], list):
            logger.warning("Outline is not a list, converting to empty list")
            json_data['outline'] = []
        
        # Validate and clean outline entries
        cleaned_outline = []
        for i, entry in enumerate(json_data['outline']):
            if not isinstance(entry, dict):
                logger.warning(f"Outline entry {i} is not a dictionary, skipping")
                continue
            
            # Ensure required fields exist
            if not all(key in entry for key in ['level', 'text', 'page']):
                logger.warning(f"Outline entry {i} missing required fields, skipping")
                continue
            
            # Validate field types and values
            try:
                entry['level'] = self.format_heading_level(entry['level'])
                entry['text'] = self.sanitize_for_json(str(entry['text']))
                entry['page'] = max(0, int(entry['page']))
                
                # Skip entries with empty text
                if not entry['text']:
                    logger.debug(f"Skipping outline entry {i} with empty text")
                    continue
                
                cleaned_outline.append(entry)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error cleaning outline entry {i}: {e}")
                continue
        
        json_data['outline'] = cleaned_outline
        return json_data
    
    def create_json_output(self, title: str, headings: List[Dict[str, Any]], 
                          validate: bool = True) -> Dict[str, Any]:
        """
        Create complete JSON output with formatting, edge case handling, and validation.
        
        Args:
            title: Document title
            headings: List of heading dictionaries
            validate: Whether to perform schema validation
            
        Returns:
            Complete JSON output ready for serialization
        """
        # Format the basic output
        json_data = self.format_output(title, headings)
        
        # Handle edge cases
        json_data = self.handle_edge_cases(json_data)
        
        # Validate against schema if requested
        if validate and not self.validate_schema(json_data):
            logger.warning("Schema validation failed, but continuing with output")
        
        return json_data
    
    def write_json_file(self, json_data: Dict[str, Any], output_path: str) -> bool:
        """
        Write JSON data to file with proper encoding.
        
        Args:
            json_data: Dictionary to write as JSON
            output_path: Path to output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, sort_keys=False)
            
            logger.info(f"Successfully wrote JSON output to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write JSON file {output_path}: {e}")
            return False
    
    def process_and_write(self, title: str, headings: List[Dict[str, Any]], 
                         output_path: str, validate: bool = True) -> bool:
        """
        Complete processing pipeline: format, validate, and write JSON output.
        
        Args:
            title: Document title
            headings: List of heading dictionaries
            output_path: Path to output file
            validate: Whether to perform schema validation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the JSON output
            json_data = self.create_json_output(title, headings, validate)
            
            # Write to file
            return self.write_json_file(json_data, output_path)
            
        except Exception as e:
            logger.error(f"Error in process_and_write: {e}")
            return False


def create_json_handler(schema_path: Optional[str] = None) -> JSONHandler:
    """
    Factory function to create a JSONHandler instance.
    
    Args:
        schema_path: Optional path to schema file
        
    Returns:
        Configured JSONHandler instance
    """
    return JSONHandler(schema_path)