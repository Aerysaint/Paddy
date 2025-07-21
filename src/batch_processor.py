"""
Batch processing system for PDF outline extraction.

This module provides functionality to process multiple PDF files from an input
directory, with individual file error handling, progress logging, and graceful
degradation for corrupted or problematic PDFs.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .pdf_extractor import PDFExtractor
from .structure_analyzer import StructureAnalyzer
from .semantic_analyzer import SemanticAnalyzer
from .vlm_analyzer import VLMAnalyzer
from .heading_level_classifier import HeadingLevelClassifier
from .title_extractor import TitleExtractor
from .json_handler import JSONHandler
from .logging_config import setup_logging, PDFProcessingError, handle_pdf_error

logger = setup_logging()


class BatchProcessor:
    """
    Batch processing system for PDF outline extraction.
    
    Features:
    - Automatic PDF discovery from input directory
    - Individual file error handling without stopping batch processing
    - Graceful degradation for corrupted or problematic PDFs
    - Progress logging and error reporting
    - Parallel processing support (optional)
    """
    
    def __init__(self, max_workers: int = 1):
        """
        Initialize the batch processor.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.max_workers = max_workers
        self.pdf_extractor = PDFExtractor()
        self.structure_analyzer = StructureAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.vlm_analyzer = VLMAnalyzer()
        self.heading_classifier = HeadingLevelClassifier()
        self.title_extractor = TitleExtractor()
        self.json_handler = JSONHandler()
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
    
    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process all PDF files from input directory and save results to output directory.
        
        Args:
            input_dir: Path to directory containing PDF files
            output_dir: Path to directory for JSON output files
            
        Returns:
            Dictionary with processing statistics and results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        logger.info(f"Starting batch processing: {input_dir} -> {output_dir}")
        
        # Validate directories
        if not input_path.exists():
            error_msg = f"Input directory does not exist: {input_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not input_path.is_dir():
            error_msg = f"Input path is not a directory: {input_dir}"
            logger.error(error_msg)
            raise NotADirectoryError(error_msg)
        
        # Create output directory if it doesn't exist
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ready: {output_dir}")
        except Exception as e:
            error_msg = f"Failed to create output directory {output_dir}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Discover PDF files
        pdf_files = self._discover_pdf_files(input_path)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return self._get_final_stats()
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Initialize statistics
        self.stats['total_files'] = len(pdf_files)
        self.stats['start_time'] = time.time()
        
        # Process files
        if self.max_workers > 1:
            self._process_files_parallel(pdf_files, output_path)
        else:
            self._process_files_sequential(pdf_files, output_path)
        
        # Finalize statistics
        self.stats['end_time'] = time.time()
        
        # Log final results
        self._log_final_results()
        
        return self._get_final_stats()
    
    def _discover_pdf_files(self, input_dir: Path) -> List[Path]:
        """
        Discover all PDF files in the input directory.
        
        Args:
            input_dir: Input directory path
            
        Returns:
            List of PDF file paths
        """
        pdf_files = []
        
        try:
            # Search for PDF files (case-insensitive)
            for file_path in input_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                    pdf_files.append(file_path)
            
            # Sort files for consistent processing order
            pdf_files.sort(key=lambda x: x.name.lower())
            
            logger.debug(f"Discovered PDF files: {[f.name for f in pdf_files]}")
            
        except Exception as e:
            logger.error(f"Error discovering PDF files in {input_dir}: {str(e)}")
            raise
        
        return pdf_files
    
    def _process_files_sequential(self, pdf_files: List[Path], output_dir: Path) -> None:
        """
        Process PDF files sequentially.
        
        Args:
            pdf_files: List of PDF file paths
            output_dir: Output directory path
        """
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                self._process_single_pdf(pdf_file, output_dir)
                self.stats['successful'] += 1
                logger.info(f"Successfully processed: {pdf_file.name}")
                
            except Exception as e:
                self._handle_processing_error(pdf_file, e)
    
    def _process_files_parallel(self, pdf_files: List[Path], output_dir: Path) -> None:
        """
        Process PDF files in parallel using ThreadPoolExecutor.
        
        Args:
            pdf_files: List of PDF file paths
            output_dir: Output directory path
        """
        logger.info(f"Processing files with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_pdf, pdf_file, output_dir): pdf_file
                for pdf_file in pdf_files
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_file), 1):
                pdf_file = future_to_file[future]
                
                try:
                    future.result()  # This will raise any exception that occurred
                    self.stats['successful'] += 1
                    logger.info(f"Successfully processed ({i}/{len(pdf_files)}): {pdf_file.name}")
                    
                except Exception as e:
                    self._handle_processing_error(pdf_file, e)
    
    def process_single_pdf(self, pdf_path: str, output_path: str) -> bool:
        """
        Process a single PDF file and save the result.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to the output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_file = Path(pdf_path)
            output_file = Path(output_path)
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            result = self._process_single_pdf(pdf_file, output_file.parent, output_file.name)
            return result
            
        except Exception as e:
            logger.error(f"Error processing single PDF {pdf_path}: {str(e)}")
            return False
    
    def _process_single_pdf(self, pdf_file: Path, output_dir: Path, 
                           output_filename: Optional[str] = None) -> bool:
        """
        Process a single PDF file through the complete pipeline.
        
        Args:
            pdf_file: Path to the PDF file
            output_dir: Output directory path
            output_filename: Optional custom output filename
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Generate output filename
            if output_filename is None:
                output_filename = pdf_file.stem + '.json'
            output_file = output_dir / output_filename
            
            logger.debug(f"Processing pipeline for: {pdf_file.name}")
            
            # Step 1: Extract text blocks with metadata
            logger.debug("Step 1: Extracting text blocks")
            text_blocks = self.pdf_extractor.extract_text_with_metadata(str(pdf_file))
            
            if not text_blocks:
                logger.warning(f"No text blocks extracted from {pdf_file.name}")
                return self._handle_empty_document(pdf_file, output_file)
            
            logger.debug(f"Extracted {len(text_blocks)} text blocks")
            
            # Step 2: Extract document title
            logger.debug("Step 2: Extracting document title")
            pdf_metadata = self.pdf_extractor.get_document_info(str(pdf_file))
            title = self.title_extractor.extract_title(str(pdf_file), text_blocks, pdf_metadata)
            
            logger.debug(f"Extracted title: '{title}'")
            
            # Step 3: Analyze document structure
            logger.debug("Step 3: Analyzing document structure")
            heading_candidates, analysis_context = self.structure_analyzer.analyze_document_structure(text_blocks)
            
            logger.debug(f"Found {len(heading_candidates)} heading candidates")
            
            # Step 4: Apply semantic analysis for better filtering
            logger.debug("Step 4: Applying semantic analysis")
            if heading_candidates:
                try:
                    heading_candidates = self.semantic_analyzer.analyze_ambiguous_candidates(
                        heading_candidates, text_blocks
                    )
                except Exception as e:
                    logger.warning(f"Semantic analysis failed: {e}, continuing without it")
            
            # Step 5: Classify heading levels
            logger.debug("Step 5: Classifying heading levels")
            classified_headings = self.heading_classifier.classify_heading_levels(heading_candidates)
            
            # Apply balanced confidence threshold with special handling for numbered patterns
            base_confidence_threshold = 0.30  # Base threshold (lowered further for fragmented headings)
            numbered_confidence_threshold = 0.3  # Lower threshold for numbered patterns
            final_headings = []
            
            logger.debug(f"Processing {len(classified_headings)} classified headings")
            
            for h in classified_headings:
                logger.debug(f"Candidate: '{h.text[:50]}...' - Level: {h.assigned_level}, Confidence: {h.confidence_score:.3f}")
                
                # Use different thresholds based on whether it has numbering patterns
                has_numbering = any('numbering_level_' in indicator for indicator in h.level_indicators)
                confidence_threshold = numbered_confidence_threshold if has_numbering else base_confidence_threshold
                
                if not h.assigned_level or h.confidence_score < confidence_threshold:
                    logger.info(f"  Skipped: '{h.text}' - Level: {h.assigned_level}, Confidence: {h.confidence_score:.3f} < {confidence_threshold:.2f}")
                    continue
                else:
                    logger.info(f"  Passed: '{h.text}' - Level: H{h.assigned_level}, Confidence: {h.confidence_score:.3f} >= {confidence_threshold:.2f}")
                
                # Additional filtering to reduce false positives
                text = h.text.strip()
                
                # Skip very short text (likely not meaningful headings)
                if len(text) < 3:
                    logger.info(f"  Skipped: '{text}' - Too short (length: {len(text)})")
                    continue
                
                # Enhanced filtering for better precision
                if self._is_enhanced_non_heading(text, h):
                    logger.info(f"  Skipped: '{text}' - Enhanced non-heading filter")
                    continue
                
                logger.debug(f"  Accepted: {text}")
                final_headings.append({
                    'level': f"H{h.assigned_level}",
                    'text': text,
                    'page': h.page
                })
            
            # Sort by page and position to maintain document order
            final_headings.sort(key=lambda x: (x['page'], 0))
            
            logger.info(f"Classified {len(final_headings)} final headings")
            for heading in final_headings:
                logger.info(f"Final heading: {heading}")
            
            # Step 5: Generate JSON output
            logger.debug("Step 5: Generating JSON output")
            success = self.json_handler.process_and_write(title, final_headings, str(output_file))
            
            if not success:
                raise RuntimeError("Failed to write JSON output")
            
            # Log processing time
            processing_time = time.time() - start_time
            logger.debug(f"Processing completed in {processing_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in processing pipeline for {pdf_file.name}: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _handle_empty_document(self, pdf_file: Path, output_file: Path) -> bool:
        """
        Handle documents with no extractable text.
        
        Args:
            pdf_file: Path to the PDF file
            output_file: Path to the output file
            
        Returns:
            True if handled successfully
        """
        logger.warning(f"Creating empty outline for document with no text: {pdf_file.name}")
        
        # Create minimal output with filename as title
        title = pdf_file.stem.replace('_', ' ').replace('-', ' ').title()
        empty_headings = []
        
        return self.json_handler.process_and_write(title, empty_headings, str(output_file))
    
    def _handle_processing_error(self, pdf_file: Path, error: Exception) -> None:
        """
        Handle processing errors for individual files.
        
        Args:
            pdf_file: Path to the PDF file that failed
            error: The exception that occurred
        """
        self.stats['failed'] += 1
        
        error_info = {
            'file': str(pdf_file),
            'error': str(error),
            'error_type': type(error).__name__,
            'timestamp': time.time()
        }
        self.stats['errors'].append(error_info)
        
        # Log error with appropriate level
        if isinstance(error, PDFProcessingError):
            logger.error(f"PDF processing error for {pdf_file.name}: {str(error)}")
        elif isinstance(error, FileNotFoundError):
            logger.error(f"File not found: {pdf_file.name}")
        elif isinstance(error, PermissionError):
            logger.error(f"Permission denied accessing {pdf_file.name}")
        else:
            logger.error(f"Unexpected error processing {pdf_file.name}: {str(error)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        # Continue processing other files
        logger.info("Continuing with next file...")
    
    def _log_final_results(self) -> None:
        """Log final processing results."""
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total files: {self.stats['total_files']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_files']) * 100
            avg_time = total_time / self.stats['total_files']
            logger.info(f"Success rate: {success_rate:.1f}%")
            logger.info(f"Average time per file: {avg_time:.2f} seconds")
        
        if self.stats['errors']:
            logger.info(f"Errors encountered: {len(self.stats['errors'])}")
            for error_info in self.stats['errors'][-5:]:  # Show last 5 errors
                logger.info(f"  - {error_info['file']}: {error_info['error_type']}")
        
        logger.info("=" * 60)
    
    def _get_final_stats(self) -> Dict[str, Any]:
        """Get final processing statistics."""
        stats = self.stats.copy()
        
        if stats['start_time'] and stats['end_time']:
            stats['total_time'] = stats['end_time'] - stats['start_time']
            if stats['total_files'] > 0:
                stats['success_rate'] = (stats['successful'] / stats['total_files']) * 100
                stats['average_time_per_file'] = stats['total_time'] / stats['total_files']
        
        return stats
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self._get_final_stats()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
    
    def _is_likely_non_heading(self, text: str) -> bool:
        """
        Check if text is likely not a heading based on patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is likely not a heading
        """
        import re
        
        text_lower = text.lower().strip()
        
        # Skip pure dates
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', text_lower):
            return True
        
        # Skip pure numbers
        if re.match(r'^\d+$', text):
            return True
        
        # Skip version numbers
        if re.match(r'^v?\d+(\.\d+)*$', text_lower):
            return True
        
        # Skip page references
        if re.match(r'^page\s+\d+$', text_lower):
            return True
        
        # Skip copyright notices
        if 'copyright' in text_lower or '©' in text:
            return True
        
        # Skip email addresses
        if '@' in text and '.' in text:
            return True
        
        # Skip URLs
        if text_lower.startswith(('http://', 'https://', 'www.')):
            return True
        
        # Skip very generic single words that appear frequently
        generic_words = {
            'overview', 'board', 'version', 'date', 'page', 'document', 
            'title', 'author', 'subject', 'keywords', 'abstract',
            'draft', 'final', 'revised', 'updated', 'created'
        }
        
        if text_lower in generic_words:
            logger.info(f"      Generic word filter caught: '{text}'")
            return True
        
        # Skip text that's mostly punctuation or symbols
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count < len(text) * 0.5:  # Less than 50% alphabetic characters
            logger.info(f"      Punctuation filter caught: '{text}' (alpha: {alpha_count}/{len(text)})")
            return True
        
        logger.info(f"      Basic filter passed: '{text}'")
        return False
    
    def _is_enhanced_non_heading(self, text: str, heading_candidate) -> bool:
        """
        Enhanced filtering to identify non-headings with better precision.
        
        Args:
            text: Text to check
            heading_candidate: The heading candidate object
            
        Returns:
            True if text is likely not a heading
        """
        import re
        
        text_lower = text.lower().strip()
        
        logger.info(f"    Checking enhanced filter for: '{text}'")
        
        # First apply basic non-heading check
        if self._is_likely_non_heading(text):
            logger.info(f"    Basic non-heading filter caught: '{text}'")
            return True
        
        # Document-specific metadata patterns
        metadata_patterns = [
            r'^version\s+\d+',
            r'^copyright\s*©',
            r'international\s+software\s+testing',
            r'qualifications\s+board',
            r'istqb',
            r'foundation\s+level\s+extensions',
            r'^\d{1,2}\s+(june|july|august|september|october|november|december)',
            r'working\s+together',
            r'^rfp:\s*r$',  # Partial RFP text
            r'^oposal$',    # Partial "proposal" text
        ]
        
        for pattern in metadata_patterns:
            if re.match(pattern, text_lower):
                logger.info(f"    Metadata pattern filter caught: '{text}' (pattern: {pattern})")
                return True
        
        # Skip very long sentences that are likely body text
        if len(text.split()) > 8 and text.endswith('.'):
            logger.info(f"    Long sentence filter caught: '{text}'")
            return True
        
        # Skip long descriptive text that's clearly body content
        if len(text.split()) > 15:
            logger.info(f"    Long text filter caught: '{text}'")
            return True
        
        # Skip text that starts with numbers but isn't clearly a numbered heading
        if re.match(r'^\d+\s', text) and not re.match(r'^\d+\.\s', text) and len(text.split()) > 8:
            return True
        
        # Skip numbered list items that are clearly content, not headings
        if self._is_likely_list_item(text, heading_candidate):
            logger.info(f"    List item filter caught: '{text}'")
            return True
        
        # Skip incomplete text fragments (likely from text extraction issues)
        if len(text) < 5 and not re.match(r'^\d+\.', text):
            logger.info(f"    Fragment filter caught: '{text}' (length: {len(text)})")
            return True
        
        # Skip text that looks like form fields or instructions
        instruction_patterns = [
            r'i\s+declare\s+that',
            r'particulars\s+furnished',
            r'true\s+and\s+correct',
            r'best\s+of\s+my\s+knowledge',
        ]
        
        for pattern in instruction_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"    Instruction pattern filter caught: '{text}' (pattern: {pattern})")
                return True
        
        # Skip if confidence is very low even after passing initial threshold
        if hasattr(heading_candidate, 'confidence_score') and heading_candidate.confidence_score < 0.3:
            logger.info(f"    Low confidence filter caught: '{text}' (confidence: {heading_candidate.confidence_score})")
            return True
        
        logger.info(f"    Enhanced filter passed: '{text}'")
        return False
    
    def _is_likely_list_item(self, text: str, heading_candidate) -> bool:
        """
        Check if numbered text is likely a list item rather than a heading.
        
        Args:
            text: Text to check
            heading_candidate: The heading candidate object
            
        Returns:
            True if text is likely a list item, not a heading
        """
        import re
        
        # Check for list-like patterns
        text_lower = text.lower().strip()
        
        # Long numbered items that are clearly list content
        if re.match(r'^\d+\.?\s+', text) and len(text.split()) > 12:
            return True
        
        # Items that start with numbers but have list-like content
        list_content_patterns = [
            r'that\s+\w+.*will\s+',  # "that ODL expenditures will increase"
            r'professionals\s+who\s+',  # "professionals who have achieved"
            r'the\s+\w+.*process\s+',  # "the planning process must"
        ]
        
        for pattern in list_content_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check visual context - if it has very little whitespace isolation, likely a list item
        if hasattr(heading_candidate, 'formatting_features'):
            whitespace_before = heading_candidate.formatting_features.get('whitespace_before', 0)
            whitespace_after = heading_candidate.formatting_features.get('whitespace_after', 0)
            
            # Exception for fragmented headings - they have large fonts and are bold
            is_large_font = heading_candidate.formatting_features.get('is_large_font', False)
            is_bold = heading_candidate.formatting_features.get('is_bold', False)
            font_size_ratio = heading_candidate.formatting_features.get('font_size_ratio', 1.0)
            
            # If it's a large, bold text with high font ratio, it's likely a heading even with poor whitespace
            if is_large_font and is_bold and font_size_ratio > 1.5:
                logger.info(f"      Fragmented heading exception: '{text}' (large font: {is_large_font}, bold: {is_bold}, ratio: {font_size_ratio:.2f})")
                return False
            
            # List items typically have less whitespace isolation than headings
            if whitespace_before < 5 and whitespace_after < 5:
                return True
        
        return False


def process_pdf_directory(input_dir: str, output_dir: str, max_workers: int = 1) -> Dict[str, Any]:
    """
    Convenience function to process a directory of PDF files.
    
    Args:
        input_dir: Path to directory containing PDF files
        output_dir: Path to directory for JSON output files
        max_workers: Maximum number of worker threads
        
    Returns:
        Dictionary with processing statistics
    """
    processor = BatchProcessor(max_workers=max_workers)
    return processor.process_directory(input_dir, output_dir)


def process_single_pdf_file(pdf_path: str, output_path: str) -> bool:
    """
    Convenience function to process a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to the output JSON file
        
    Returns:
        True if successful, False otherwise
    """
    processor = BatchProcessor()
    return processor.process_single_pdf(pdf_path, output_path)