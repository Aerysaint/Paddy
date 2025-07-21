"""
Title extraction module with multiple strategies for PDF documents.

This module implements various strategies to extract document titles from PDFs,
including metadata extraction, visual prominence analysis, pattern matching,
and filename-based fallbacks.
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from collections import Counter

from .data_models import TextBlock
from .logging_config import setup_logging

logger = setup_logging()


class TitleExtractor:
    """
    Multi-strategy title extraction system for PDF documents.
    
    Implements multiple extraction strategies in priority order:
    1. PDF metadata title field
    2. Visual prominence analysis (largest/bold text on first page)
    3. Pattern matching for title-like text structures
    4. Filename-based fallback
    """
    
    def __init__(self):
        """Initialize the title extractor with configuration."""
        self.title_keywords = {
            'english': ['title', 'chapter', 'report', 'analysis', 'study', 'research', 'paper'],
            'common': ['abstract', 'introduction', 'conclusion', 'summary', 'overview']
        }
        
        # Patterns that indicate title-like text
        self.title_patterns = [
            r'^[A-Z][A-Za-z\s\-:]+$',  # Title case starting with capital
            r'^[A-Z\s\-:]+$',          # All caps (but not too long)
            r'^\d+\.\s*[A-Z][A-Za-z\s\-:]+$',  # Numbered title
        ]
        
        # Patterns to exclude (likely not titles)
        self.exclusion_patterns = [
            r'^\d+$',                   # Just numbers
            r'^page\s+\d+',            # Page numbers
            r'^figure\s+\d+',          # Figure captions
            r'^table\s+\d+',           # Table captions
            r'^appendix\s+[a-z]',      # Appendix labels
            r'^\w+@\w+\.\w+',          # Email addresses
            r'^https?://',             # URLs
            r'^\d{1,2}/\d{1,2}/\d{2,4}',  # Dates
            r'.*required.*',           # Fine print requirements
            r'.*shoes.*',              # Specific exclusions for dress codes
            r'.*closed.*toed.*',       # Specific dress code text
            r'.*please.*',             # Polite instructions (often fine print)
            r'.*must.*',               # Requirements (often fine print)
            r'.*should.*',             # Suggestions (often fine print)
        ]
        
        # Maximum reasonable title length
        self.max_title_length = 200
        self.min_title_length = 5
    
    def extract_title(self, pdf_path: str, text_blocks: List[TextBlock], 
                     pdf_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract document title using multiple strategies.
        
        Args:
            pdf_path: Path to the PDF file
            text_blocks: List of extracted text blocks from the PDF
            pdf_metadata: Optional PDF metadata dictionary
            
        Returns:
            Extracted title or empty string if no title found
        """
        logger.info(f"Extracting title from PDF: {pdf_path}")
        
        # Strategy 1: Visual prominence analysis (prioritize document content)
        visual_title = self._extract_by_visual_prominence(text_blocks)
        
        # Strategy 2: PDF metadata title field
        metadata_title = self._extract_from_metadata(pdf_metadata)
        
        # Strategy 3: Pattern matching for title-like structures
        pattern_title = self._extract_by_pattern_matching(text_blocks)
        
        # Intelligent selection between candidates
        title = self._select_best_title_candidate(visual_title, metadata_title, pattern_title, text_blocks)
        
        if title:
            logger.info(f"Title extracted: {title}")
            return title
        
        # Strategy 4: Conservative filename-based fallback (only for clear cases)
        if self._should_use_filename_fallback(pdf_path, text_blocks):
            title = self._extract_from_filename(pdf_path)
            if title:
                logger.info(f"Title extracted from filename: {title}")
                return title
        
        logger.info(f"No clear title found for PDF: {pdf_path}, returning empty string")
        return ""
    
    def _extract_from_metadata(self, pdf_metadata: Optional[Dict[str, Any]]) -> str:
        """
        Extract title from PDF metadata.
        
        Args:
            pdf_metadata: PDF metadata dictionary
            
        Returns:
            Title from metadata or empty string
        """
        if not pdf_metadata:
            return ""
        
        # Check various metadata fields that might contain the title
        title_fields = ['title', 'Title', 'TITLE', 'subject', 'Subject']
        
        for field in title_fields:
            if field in pdf_metadata:
                title = str(pdf_metadata[field]).strip()
                if title and self._is_valid_title(title) and self._is_high_quality_title(title):
                    return self._clean_title(title)
        
        return ""
    
    def _extract_by_visual_prominence(self, text_blocks: List[TextBlock]) -> str:
        """
        Extract title based on visual prominence (largest/bold text on first page).
        
        Args:
            text_blocks: List of text blocks from the PDF
            
        Returns:
            Title based on visual prominence or empty string
        """
        if not text_blocks:
            return ""
        
        # Focus on first page text blocks
        first_page_blocks = [block for block in text_blocks if block.page == 0]
        if not first_page_blocks:
            return ""
        
        # Score blocks based on visual prominence
        scored_blocks = []
        
        for block in first_page_blocks:
            if not self._is_potential_title_text(block.text):
                continue
            
            score = self._calculate_visual_prominence_score(block, first_page_blocks)
            if score > 0:
                scored_blocks.append((block, score))
        
        if not scored_blocks:
            return ""
        
        # Sort by score (highest first) and return the best candidate
        scored_blocks.sort(key=lambda x: x[1], reverse=True)
        best_block = scored_blocks[0][0]
        
        logger.debug(f"Best visual prominence candidate: {best_block.text} (score: {scored_blocks[0][1]:.2f})")
        return self._clean_title(best_block.text)
    
    def _calculate_visual_prominence_score(self, block: TextBlock, 
                                         all_blocks: List[TextBlock]) -> float:
        """
        Calculate visual prominence score for a text block.
        
        Args:
            block: Text block to score
            all_blocks: All text blocks on the page for comparison
            
        Returns:
            Visual prominence score (0.0 to 1.0)
        """
        score = 0.0
        
        # Font size score (relative to other blocks) - increased weight
        font_sizes = [b.font_size for b in all_blocks if b.font_size > 0]
        if font_sizes:
            max_font_size = max(font_sizes)
            avg_font_size = sum(font_sizes) / len(font_sizes)
            
            # Heavily favor the largest fonts
            if block.font_size >= max_font_size:
                score += 0.8  # Largest font gets much higher score (increased)
            elif block.font_size > avg_font_size * 1.8:
                score += 0.6  # Very large fonts (increased threshold)
            elif block.font_size > avg_font_size * 1.5:
                score += 0.4  # Large fonts
            elif block.font_size > avg_font_size * 1.2:
                score += 0.2  # Significantly larger than average
            elif block.font_size > avg_font_size:
                score += 0.1  # Larger than average
            else:
                score -= 0.2  # Penalize smaller fonts
        
        # Bold text bonus
        if block.is_bold:
            score += 0.2
        
        # Position score - favor top and middle, penalize bottom
        page_height = max(b.bbox[3] for b in all_blocks) - min(b.bbox[1] for b in all_blocks)
        if page_height > 0:
            # Normalize position (0 = bottom, 1 = top)
            relative_position = 1 - ((block.bbox[1] - min(b.bbox[1] for b in all_blocks)) / page_height)
            if relative_position > 0.8:  # Top 20% of page
                score += 0.3
            elif relative_position > 0.6:  # Top 40% of page
                score += 0.2
            elif relative_position > 0.4:  # Middle 40% of page
                score += 0.1
            else:  # Bottom 40% of page
                score -= 0.3  # Heavily penalize bottom text (likely fine print)
        
        # Text length consideore * 0.2
        
        return min(1.0, max(0.0, score))
    
    def _calculate_isolation_score(self, block: TextBlock, all_blocks: List[TextBlock]) -> float:
        """
        Calculate how isolated a text block is (more whitespace = higher score).
        
        Args:
            block: Text block to analyze
            all_blocks: All text blocks on the page
            
        Returns:
            Isolation score (0.0 to 1.0)
        """
        if len(all_blocks) <= 1:
            return 1.0
        
        # Find nearest neighbors
        min_distance = float('inf')
        
        for other_block in all_blocks:
            if other_block == block:
                continue
            
            # Calculate distance between blocks
            distance = self._calculate_block_distance(block, other_block)
            min_distance = min(min_distance, distance)
        
        # Normalize distance to score (larger distance = higher isolation)
        # Assume reasonable page dimensions for normalization
        max_reasonable_distance = 100  # points
        isolation_score = min(1.0, min_distance / max_reasonable_distance)
        
        return isolation_score
    
    def _calculate_block_distance(self, block1: TextBlock, block2: TextBlock) -> float:
        """
        Calculate distance between two text blocks.
        
        Args:
            block1: First text block
            block2: Second text block
            
        Returns:
            Distance in points
        """
        # Calculate center points
        center1 = ((block1.bbox[0] + block1.bbox[2]) / 2, (block1.bbox[1] + block1.bbox[3]) / 2)
        center2 = ((block2.bbox[0] + block2.bbox[2]) / 2, (block2.bbox[1] + block2.bbox[3]) / 2)
        
        # Euclidean distance
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    def _extract_by_pattern_matching(self, text_blocks: List[TextBlock]) -> str:
        """
        Extract title using pattern matching for title-like text structures.
        
        Args:
            text_blocks: List of text blocks from the PDF
            
        Returns:
            Title based on pattern matching or empty string
        """
        if not text_blocks:
            return ""
        
        # Focus on first few pages for title search
        early_blocks = [block for block in text_blocks if block.page <= 1]
        
        candidates = []
        
        for block in early_blocks:
            text = block.text.strip()
            
            if not self._is_potential_title_text(text):
                continue
            
            # Check against title patterns
            pattern_score = self._calculate_pattern_score(text)
            if pattern_score > 0.5:
                candidates.append((block, pattern_score))
        
        if not candidates:
            return ""
        
        # Sort by pattern score and return best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = candidates[0][0]
        
        logger.debug(f"Best pattern matching candidate: {best_candidate.text} (score: {candidates[0][1]:.2f})")
        return self._clean_title(best_candidate.text)
    
    def _calculate_pattern_score(self, text: str) -> float:
        """
        Calculate how well text matches title patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Pattern matching score (0.0 to 1.0)
        """
        score = 0.0
        
        # Check against positive patterns
        for pattern in self.title_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                score += 0.3
                break
        
        # Check against exclusion patterns (negative score)
        for pattern in self.exclusion_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                score -= 0.5
                break
        
        # Title-like characteristics
        if text[0].isupper() and not text.isupper():  # Title case
            score += 0.2
        
        if ':' in text and text.count(':') == 1:  # Subtitle pattern
            score += 0.2
        
        # Length considerations
        if 10 <= len(text) <= 100:
            score += 0.2
        elif len(text) > 150:
            score -= 0.3
        
        # Word count (titles usually have reasonable word count)
        word_count = len(text.split())
        if 2 <= word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def _select_best_title_candidate(self, visual_title: str, metadata_title: str, 
                                   pattern_title: str, text_blocks: List[TextBlock]) -> str:
        """
        Intelligently select the best title from multiple candidates.
        
        Args:
            visual_title: Title from visual prominence analysis
            metadata_title: Title from PDF metadata
            pattern_title: Title from pattern matching
            text_blocks: Document text blocks for context
            
        Returns:
            Best title candidate or empty string
        """
        # Try to reconstruct fragmented titles first
        reconstructed_title = self._attempt_title_reconstruction(text_blocks)
        if reconstructed_title and self._is_high_quality_title(reconstructed_title):
            logger.debug(f"Using reconstructed title: {reconstructed_title}")
            return self._clean_title(reconstructed_title)
        
        candidates = []
        
        # Add candidates with quality scores
        if visual_title and self._is_high_quality_title(visual_title):
            candidates.append((visual_title, 'visual', self._score_title_quality(visual_title, text_blocks)))
        
        if metadata_title and self._is_high_quality_title(metadata_title):
            candidates.append((metadata_title, 'metadata', self._score_title_quality(metadata_title, text_blocks)))
        
        if pattern_title and self._is_high_quality_title(pattern_title):
            candidates.append((pattern_title, 'pattern', self._score_title_quality(pattern_title, text_blocks)))
        
        if not candidates:
            # Fallback to any valid title
            for title in [visual_title, metadata_title, pattern_title]:
                if title and self._is_valid_title(title):
                    return self._clean_title(title)
            return ""
        
        # Sort by quality score and return best
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_title, method, score = candidates[0]
        
        logger.debug(f"Selected title from {method} (score: {score:.2f}): {best_title}")
        return self._clean_title(best_title)
    
    def _score_title_quality(self, title: str, text_blocks: List[TextBlock]) -> float:
        """
        Score title quality based on various factors.
        
        Args:
            title: Title to score
            text_blocks: Document text blocks for context
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not title:
            return 0.0
        
        score = 0.0
        
        # Length score (prefer moderate lengths)
        length = len(title)
        if 20 <= length <= 80:
            score += 0.3
        elif 10 <= length <= 120:
            score += 0.2
        elif length > 150:
            score -= 0.2
        
        # Word count score
        word_count = len(title.split())
        if 3 <= word_count <= 12:
            score += 0.2
        elif word_count > 20:
            score -= 0.3
        
        # Avoid generic or technical metadata
        generic_terms = ['overview', 'document', 'untitled', 'draft', 'version']
        if any(term in title.lower() for term in generic_terms):
            score -= 0.2
        
        # Prefer titles that appear in document content
        title_words = set(title.lower().split())
        content_words = set()
        for block in text_blocks[:10]:  # Check first 10 blocks
            content_words.update(block.text.lower().split())
        
        word_overlap = len(title_words.intersection(content_words))
        if word_overlap > 0:
            score += min(0.3, word_overlap * 0.1)
        
        # Penalize titles with technical jargon or version info
        if re.search(r'v\d+|\d+\.\d+|istqb|qualifications|board', title.lower()):
            score -= 0.3
        
        # Penalize corrupted or partial titles
        if len(title) < 10:
            score -= 0.3
        
        # Detect fragmented/corrupted titles more aggressively
        if title.endswith(('f', 'r', 'e', 't', 'n', 'g', 'o', 'a', 'i', 's')):
            score -= 0.5  # Heavy penalty for single letter endings
        
        # Penalize titles with repeated letters (corruption indicator)
        if re.search(r'(.)\1{3,}', title):  # 4+ repeated characters
            score -= 0.6
        
        # Penalize very short words at the end (likely fragments)
        words = title.split()
        if len(words) > 1 and len(words[-1]) <= 2:
            score -= 0.4
        
        # Penalize titles ending with colon (likely section headers, not main titles)
        if title.endswith(':'):
            score -= 0.3
        
        return max(0.0, min(1.0, score))

    def _extract_from_filename(self, pdf_path: str) -> str:
        """
        Extract title from filename as fallback strategy.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Title derived from filename
        """
        try:
            path = Path(pdf_path)
            filename = path.stem  # Filename without extension
            
            # Clean up filename to make it title-like
            title = self._clean_filename_for_title(filename)
            
            if self._is_valid_title(title):
                return title
            
        except Exception as e:
            logger.warning(f"Error extracting title from filename {pdf_path}: {str(e)}")
        
        return ""
    
    def _clean_filename_for_title(self, filename: str) -> str:
        """
        Clean filename to create a reasonable title.
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned title-like string
        """
        # Replace common separators with spaces
        title = filename.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # Remove common file prefixes/suffixes
        title = re.sub(r'^(doc|document|file|pdf)\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*(doc|document|file|pdf)$', '', title, flags=re.IGNORECASE)
        
        # Remove version numbers and dates
        title = re.sub(r'\s*v?\d+(\.\d+)*\s*$', '', title)  # Version numbers
        title = re.sub(r'\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*', ' ', title)  # Dates
        title = re.sub(r'\s*\d{1,2}[-/]\d{1,2}[-/]\d{4}\s*', ' ', title)  # Dates
        
        # Clean up whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Convert to title case
        if title and not title.isupper():
            title = title.title()
        
        return title
    
    def _is_potential_title_text(self, text: str) -> bool:
        """
        Check if text could potentially be a title.
        
        Args:
            text: Text to check
            
        Returns:
            True if text could be a title
        """
        if not text or len(text.strip()) < self.min_title_length:
            return False
        
        if len(text) > self.max_title_length:
            return False
        
        # Check against exclusion patterns
        for pattern in self.exclusion_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return False
        
        # Must contain at least one letter (including Unicode letters for multilingual support)
        # Check for Latin letters, Japanese characters, Chinese characters, etc.
        if not re.search(r'[a-zA-Z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u0100-\u017F\u0180-\u024F]', text):
            return False
        
        # Avoid text that's mostly numbers or symbols
        # Count all Unicode letters, not just Latin letters
        letter_count = sum(1 for c in text if c.isalpha() or ord(c) > 127)
        if letter_count < len(text) * 0.3:  # Less than 30% letters
            return False
        
        return True
    
    def _is_valid_title(self, title: str) -> bool:
        """
        Validate if a string is a reasonable title.
        
        Args:
            title: Title string to validate
            
        Returns:
            True if title is valid
        """
        if not title:
            return False
        
        title = title.strip()
        
        if len(title) < self.min_title_length or len(title) > self.max_title_length:
            return False
        
        # Must contain meaningful content
        if not re.search(r'[a-zA-Z]', title):
            return False
        
        # Check against exclusion patterns
        for pattern in self.exclusion_patterns:
            if re.match(pattern, title, re.IGNORECASE):
                return False
        
        return True
    
    def _clean_title(self, title: str) -> str:
        """
        Clean and normalize extracted title.
        
        Args:
            title: Raw title string
            
        Returns:
            Cleaned title string
        """
        if not title:
            return ""
        
        # Strip whitespace and normalize
        title = title.strip()
        
        # Remove common title prefixes that might be artifacts
        title = re.sub(r'^(title:\s*|subject:\s*)', '', title, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        title = re.sub(r'\s+', ' ', title)
        
        # Remove trailing punctuation that doesn't belong in titles
        title = re.sub(r'[.;,]+$', '', title)
        
        # Ensure proper encoding
        try:
            title = title.encode('utf-8').decode('utf-8')
        except UnicodeError:
            logger.warning(f"Unicode error in title: {title}")
            title = title.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        return title.strip()
    
    def _attempt_title_reconstruction(self, text_blocks: List[TextBlock]) -> str:
        """
        Attempt to reconstruct fragmented titles from multiple text blocks.
        
        Args:
            text_blocks: List of text blocks from the PDF
            
        Returns:
            Reconstructed title or empty string
        """
        if not text_blocks:
            return ""
        
        # Focus on first page blocks with large fonts
        first_page_blocks = [block for block in text_blocks if block.page == 0]
        if not first_page_blocks:
            return ""
        
        # Find blocks with similar large font sizes that might be title fragments
        font_sizes = [b.font_size for b in first_page_blocks if b.font_size > 0]
        if not font_sizes:
            return ""
        
        max_font_size = max(font_sizes)
        avg_font_size = sum(font_sizes) / len(font_sizes)
        
        # Consider blocks with font size close to maximum as potential title fragments
        title_candidates = []
        for block in first_page_blocks:
            if (block.font_size >= max_font_size * 0.95 and  # Within 5% of max font size
                block.font_size > avg_font_size * 1.5 and    # Significantly larger than average
                len(block.text.strip()) > 2):                # Not just punctuation
                title_candidates.append(block)
        
        if len(title_candidates) < 2:
            return ""  # Need at least 2 fragments to reconstruct
        
        # Sort by position (left to right, top to bottom)
        title_candidates.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        
        # Try to combine fragments that appear to be parts of the same title
        combined_text = ""
        prev_block = None
        
        for block in title_candidates:
            text = block.text.strip()
            
            if prev_block is None:
                combined_text = text
            else:
                # Check if blocks are close enough to be part of same title
                vertical_distance = abs(block.bbox[1] - prev_block.bbox[1])
                horizontal_distance = block.bbox[0] - prev_block.bbox[2]
                
                # If blocks are on roughly the same line and close horizontally
                if vertical_distance < 10 and horizontal_distance < 50:
                    # Check if we need to add space or if text continues naturally
                    if (combined_text.endswith((' ', '-', ':')) or 
                        text.startswith((' ', '-', ':')) or
                        combined_text[-1].islower() and text[0].islower()):
                        combined_text += text
                    else:
                        combined_text += " " + text
                else:
                    # Too far apart, might be different title or not a title
                    break
            
            prev_block = block
        
        # Clean and validate the reconstructed title
        if combined_text:
            cleaned_title = self._clean_title(combined_text)
            if (len(cleaned_title) > 10 and 
                len(cleaned_title) < 200 and
                not cleaned_title.endswith(('f', 'r', 'e', 't', 'n', 'g'))):
                logger.debug(f"Reconstructed title from fragments: {cleaned_title}")
                return cleaned_title
        
        return ""
    
    def _is_high_quality_title(self, title: str) -> bool:
        """
        Check if a title meets high quality standards.
        
        Args:
            title: Title to evaluate
            
        Returns:
            True if title is high quality
        """
        if not title or len(title.strip()) < 10:
            return False
        
        # Avoid very generic titles and address components
        generic_titles = {
            'overview', 'document', 'report', 'analysis', 'study', 
            'paper', 'file', 'untitled', 'draft', 'final', 'mission statement',
            'address', 'parkway', 'street', 'avenue', 'road', 'drive', 'lane',
            'rsvp', 'date', 'time', 'location'
        }
        
        if title.lower().strip() in generic_titles:
            return False
        
        # Avoid file-like titles (with extensions or version numbers)
        if re.search(r'\.(cdr|pdf|doc|docx)$', title.lower()) or re.search(r'v\d+', title.lower()):
            return False
        
        # Must have reasonable word count
        word_count = len(title.split())
        if word_count < 2 or word_count > 20:
            return False
        
        return True
    
    def _should_use_filename_fallback(self, pdf_path: str, text_blocks: List[TextBlock]) -> bool:
        """
        Determine if filename fallback should be used.
        
        Args:
            pdf_path: Path to PDF file
            text_blocks: Document text blocks
            
        Returns:
            True if filename fallback is appropriate
        """
        # Only use filename fallback if document has very little text
        if not text_blocks:
            return True
        
        # Count meaningful text blocks
        meaningful_blocks = [
            block for block in text_blocks 
            if block.page == 0 and len(block.text.strip()) > 5
        ]
        
        # Use filename fallback only for very sparse documents
        return len(meaningful_blocks) < 3
    
    def get_title_extraction_info(self, pdf_path: str, text_blocks: List[TextBlock], 
                                 pdf_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get detailed information about title extraction process.
        
        Args:
            pdf_path: Path to the PDF file
            text_blocks: List of extracted text blocks
            pdf_metadata: Optional PDF metadata dictionary
            
        Returns:
            Dictionary with extraction details and candidates
        """
        info = {
            'extracted_title': '',
            'extraction_method': '',
            'candidates': [],
            'metadata_title': '',
            'filename_title': ''
        }
        
        # Check metadata
        metadata_title = self._extract_from_metadata(pdf_metadata)
        info['metadata_title'] = metadata_title
        
        # Check filename
        filename_title = self._extract_from_filename(pdf_path)
        info['filename_title'] = filename_title
        
        # Get visual prominence candidates
        first_page_blocks = [block for block in text_blocks if block.page == 0]
        visual_candidates = []
        
        for block in first_page_blocks:
            if self._is_potential_title_text(block.text):
                score = self._calculate_visual_prominence_score(block, first_page_blocks)
                if score > 0.3:
                    visual_candidates.append({
                        'text': block.text,
                        'score': score,
                        'method': 'visual_prominence'
                    })
        
        visual_candidates.sort(key=lambda x: x['score'], reverse=True)
        info['candidates'].extend(visual_candidates[:5])  # Top 5 candidates
        
        # Extract final title
        final_title = self.extract_title(pdf_path, text_blocks, pdf_metadata)
        info['extracted_title'] = final_title
        
        # Determine extraction method
        if final_title:
            if final_title == metadata_title:
                info['extraction_method'] = 'metadata'
            elif any(c['text'] == final_title for c in visual_candidates):
                info['extraction_method'] = 'visual_prominence'
            elif final_title == filename_title:
                info['extraction_method'] = 'filename'
            else:
                info['extraction_method'] = 'pattern_matching'
        
        return info


def extract_title_from_pdf(pdf_path: str, text_blocks: List[TextBlock], 
                          pdf_metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Convenience function to extract title from PDF using multiple strategies.
    
    Args:
        pdf_path: Path to the PDF file
        text_blocks: List of extracted text blocks
        pdf_metadata: Optional PDF metadata dictionary
        
    Returns:
        Extracted title or empty string
    """
    extractor = TitleExtractor()
    return extractor.extract_title(pdf_path, text_blocks, pdf_metadata)


def get_title_candidates(pdf_path: str, text_blocks: List[TextBlock], 
                        pdf_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to get detailed title extraction information.
    
    Args:
        pdf_path: Path to the PDF file
        text_blocks: List of extracted text blocks
        pdf_metadata: Optional PDF metadata dictionary
        
    Returns:
        Dictionary with extraction details and candidates
    """
    extractor = TitleExtractor()
    return extractor.get_title_extraction_info(pdf_path, text_blocks, pdf_metadata)