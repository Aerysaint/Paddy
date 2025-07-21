"""
PDF text extraction module with PyMuPDF integration.

This module provides functionality to extract text blocks from PDF files
with rich formatting metadata including font information, positioning,
and special character preservation.
"""

import fitz  # PyMuPDF
import unicodedata
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .data_models import TextBlock
from .logging_config import setup_logging

logger = setup_logging()


class PDFExtractor:
    """
    PDF text extraction class with comprehensive formatting metadata extraction.
    
    Handles UTF-8 encoding, special character preservation, and provides
    detailed font and positioning information for each text block.
    """
    
    def __init__(self):
        """Initialize the PDF extractor."""
        self.supported_extensions = {'.pdf'}
        self.encoding_fallbacks = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    def extract_text_with_metadata(self, pdf_path: str) -> List[TextBlock]:
        """
        Extract text blocks with comprehensive formatting metadata from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of TextBlock objects with formatting metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a valid PDF
            Exception: For other PDF processing errors
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {pdf_path.suffix}")
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            # Open PDF document
            doc = fitz.open(str(pdf_path))
            text_blocks = []
            page_count = doc.page_count  # Store page count before processing
            
            for page_num in range(page_count):
                page = doc[page_num]
                page_blocks = self._extract_page_text_blocks(page, page_num)  # Use 0-based page numbering
                text_blocks.extend(page_blocks)
                
                logger.debug(f"Extracted {len(page_blocks)} text blocks from page {page_num}")
            
            doc.close()
            
            # Post-process text blocks for normalization and validation
            text_blocks = self._post_process_text_blocks(text_blocks)
            
            logger.info(f"Successfully extracted {len(text_blocks)} text blocks from {page_count} pages")
            return text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise
    
    def _extract_page_text_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """
        Extract text blocks from a single PDF page with formatting metadata.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (1-indexed)
            
        Returns:
            List of TextBlock objects for the page
        """
        text_blocks = []
        
        try:
            # Get text blocks with formatting information
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" not in block:
                    continue  # Skip non-text blocks (images, etc.)
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Extract text and normalize
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        # Preserve special characters and normalize Unicode
                        text = self._normalize_text(text)
                        
                        # Extract font metadata
                        font_info = self._extract_font_info(span)
                        
                        # Create text block
                        text_block = TextBlock(
                            text=text,
                            page=page_num,
                            font_size=font_info["size"],
                            font_name=font_info["name"],
                            is_bold=font_info["is_bold"],
                            bbox=tuple(span["bbox"]),
                            line_height=self._calculate_line_height(span, line)
                        )
                        
                        text_blocks.append(text_block)
                        
        except Exception as e:
            logger.warning(f"Error extracting text blocks from page {page_num}: {str(e)}")
        
        return text_blocks
    
    def _extract_font_info(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract font information from a text span.
        
        Args:
            span: PyMuPDF text span dictionary
            
        Returns:
            Dictionary with font information
        """
        font_name = span.get("font", "Unknown")
        font_size = span.get("size", 12.0)
        font_flags = span.get("flags", 0)
        
        # Determine if text is bold based on font flags or font name
        is_bold = bool(font_flags & 2**4) or "bold" in font_name.lower()
        
        # Clean font name
        font_name = self._clean_font_name(font_name)
        
        return {
            "name": font_name,
            "size": float(font_size),
            "is_bold": is_bold,
            "flags": font_flags
        }
    
    def _clean_font_name(self, font_name: str) -> str:
        """
        Clean and normalize font name.
        
        Args:
            font_name: Raw font name from PDF
            
        Returns:
            Cleaned font name
        """
        if not font_name:
            return "Unknown"
        
        # Remove common prefixes and suffixes
        font_name = font_name.replace("ABCDEE+", "").replace("BCDFEE+", "")
        
        # Handle embedded font names with random prefixes
        if "+" in font_name and len(font_name.split("+")[0]) <= 6:
            font_name = font_name.split("+", 1)[1]
        
        return font_name.strip()
    
    def _calculate_line_height(self, span: Dict[str, Any], line: Dict[str, Any]) -> float:
        """
        Calculate line height for a text span.
        
        Args:
            span: PyMuPDF text span dictionary
            line: PyMuPDF line dictionary
            
        Returns:
            Line height in points
        """
        # Use span bbox height as primary indicator
        span_height = span["bbox"][3] - span["bbox"][1]
        
        # Use line bbox as fallback
        line_height = line["bbox"][3] - line["bbox"][1]
        
        # Return the larger of the two, with minimum threshold
        return max(span_height, line_height, span.get("size", 12.0) * 1.2)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving special characters and formatting.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Normalized text with preserved special characters
        """
        if not text:
            return ""
        
        # Apply Unicode NFC normalization to handle composed vs decomposed characters
        text = unicodedata.normalize('NFC', text)
        
        # Preserve special characters but clean up whitespace
        # Replace multiple whitespace with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove zero-width characters that might interfere with processing
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        text = text.replace('\ufeff', '')  # Byte order mark
        
        return text.strip()
    
    def _post_process_text_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Post-process extracted text blocks for validation and cleanup.
        
        Args:
            text_blocks: List of extracted text blocks
            
        Returns:
            Processed and validated text blocks
        """
        processed_blocks = []
        
        for block in text_blocks:
            # Skip empty blocks
            if block.is_empty():
                continue
            
            # Validate and fix any data issues
            try:
                # Ensure text is properly encoded
                block.text = self._ensure_utf8_encoding(block.text)
                
                # Validate numeric values
                if block.font_size <= 0:
                    block.font_size = 12.0
                    logger.debug(f"Fixed invalid font size for text: {block.text[:50]}...")
                
                if block.line_height <= 0:
                    block.line_height = block.font_size * 1.2
                
                processed_blocks.append(block)
                
            except Exception as e:
                logger.warning(f"Error processing text block '{block.text[:50]}...': {str(e)}")
                continue
        
        return processed_blocks
    
    def _ensure_utf8_encoding(self, text: str) -> str:
        """
        Ensure text is properly UTF-8 encoded.
        
        Args:
            text: Input text
            
        Returns:
            UTF-8 encoded text
        """
        if not text:
            return ""
        
        try:
            # Try to encode/decode to ensure valid UTF-8
            text.encode('utf-8').decode('utf-8')
            return text
        except UnicodeError:
            # Try fallback encodings
            for encoding in self.encoding_fallbacks:
                try:
                    if isinstance(text, bytes):
                        return text.decode(encoding)
                    else:
                        return text.encode(encoding, errors='ignore').decode('utf-8', errors='ignore')
                except (UnicodeError, AttributeError):
                    continue
            
            # Last resort: remove problematic characters
            logger.warning(f"Could not properly encode text, removing problematic characters: {text[:50]}...")
            return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    
    def get_document_title(self, pdf_path: str) -> str:
        """
        Extract document title using multiple strategies.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Document title or empty string if not found
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.warning(f"PDF file not found for title extraction: {pdf_path}")
            return ""
        
        try:
            doc = fitz.open(str(pdf_path))
            title = ""
            
            # Strategy 1: PDF metadata title field
            metadata = doc.metadata
            if metadata and metadata.get("title"):
                title = metadata["title"].strip()
                logger.debug(f"Found title in metadata: {title}")
            
            doc.close()
            
            # Normalize and validate title
            if title:
                title = self._normalize_text(title)
                title = self._ensure_utf8_encoding(title)
            
            return title
            
        except Exception as e:
            logger.warning(f"Error extracting title from PDF {pdf_path}: {str(e)}")
            return ""
    
    def get_document_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get comprehensive document information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with document information
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            return {}
        
        try:
            doc = fitz.open(str(pdf_path))
            page_count = doc.page_count  # Store page count before closing
            metadata = doc.metadata or {}
            doc.close()
            
            info = {
                "page_count": page_count,
                "title": self.get_document_title(str(pdf_path)),
                "metadata": metadata,
                "file_size": pdf_path.stat().st_size,
                "file_name": pdf_path.name
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting document info for {pdf_path}: {str(e)}")
            return {"file_name": pdf_path.name, "error": str(e)}


def extract_pdf_text(pdf_path: str) -> List[TextBlock]:
    """
    Convenience function to extract text blocks from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of TextBlock objects
    """
    extractor = PDFExtractor()
    return extractor.extract_text_with_metadata(pdf_path)


def get_pdf_title(pdf_path: str) -> str:
    """
    Convenience function to extract title from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Document title or empty string
    """
    extractor = PDFExtractor()
    return extractor.get_document_title(pdf_path)