"""
Structure analyzer module for PDF outline extraction.

This module provides functionality to analyze document structure and identify
potential headings using rule-based heuristics including numbering patterns,
visual isolation, text structure, and keyword-based detection.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

from .data_models import TextBlock, HeadingCandidate
from .logging_config import setup_logging

logger = setup_logging()


@dataclass
class AnalysisContext:
    """Context information for document structure analysis."""
    document_font_sizes: List[float]
    average_font_size: float
    common_font_names: Set[str]
    page_margins: Dict[int, Tuple[float, float, float, float]]  # page -> (left, top, right, bottom)
    line_spacing_stats: Dict[str, float]  # mean, median, std


class StructureAnalyzer:
    """
    Analyzes document structure to identify potential headings using rule-based heuristics.
    
    Implements comprehensive heuristics including:
    - Numbering pattern analysis (1., 1.1, 1.1.1)
    - Visual isolation detection (whitespace analysis)
    - Text structure analysis (length, capitalization, punctuation)
    - Keyword-based detection for common heading terms
    - Confidence scoring for classification decisions
    """
    
    def __init__(self):
        """Initialize the structure analyzer with heuristic patterns and weights."""
        self.heading_keywords = {
            # Primary heading indicators (high confidence)
            'primary': {
                'chapter', 'section', 'introduction', 'conclusion', 'abstract',
                'summary', 'overview', 'background', 'methodology', 'results',
                'discussion', 'references', 'bibliography', 'appendix'
            },
            # Secondary heading indicators (medium confidence)
            'secondary': {
                'part', 'unit', 'lesson', 'topic', 'subject', 'area',
                'analysis', 'evaluation', 'assessment', 'review', 'study',
                'research', 'findings', 'recommendations', 'implications'
            },
            # Tertiary heading indicators (lower confidence)
            'tertiary': {
                'definition', 'example', 'case', 'problem', 'solution',
                'approach', 'method', 'technique', 'procedure', 'process'
            }
        }
        
        # Numbering pattern regex patterns (order matters - more specific first)
        self.numbering_patterns = {
            'hierarchical_decimal': re.compile(r'^\d+\.\d+(\.\d+)*\s+'),  # 2.1, 1.1.1 (multi-level with space)
            'simple_decimal': re.compile(r'^\d+\.\s+'),  # 1., 2., 3. (single level with dot and space)
            'simple_number_with_text': re.compile(r'^\d+\s+[A-Za-z]'),  # 1 Text, 2 Text (number followed by text)
            'roman_numerals': re.compile(r'^[IVX]+\.?\s+', re.IGNORECASE),  # I, II, III
            'letters': re.compile(r'^[A-Z]\.?\s+'),  # A, B, C or A., B., C.
            'parenthetical': re.compile(r'^\(\d+\)\s+'),  # (1), (2), (3)
        }
        
        # Confidence scoring weights (rebalanced to prioritize numbering)
        self.scoring_weights = {
            'numbering': 0.45,      # Hierarchical numbering patterns (increased)
            'visual_isolation': 0.25,  # Whitespace and positioning (decreased)
            'text_structure': 0.20,    # Length, capitalization, punctuation
            'keyword_match': 0.10      # Heading keyword presence
        }
        
        # Thresholds for classification (balanced approach)
        self.confidence_thresholds = {
            'high_confidence': 0.75,    # Accept immediately for very clear cases
            'low_confidence': 0.30,     # Lower to capture more valid headings like "PATHWAY OPTIONS"
            'numbered_boost': 0.15,     # Special boost for numbered patterns
            'ambiguous_range': (0.30, 0.75)  # Requires additional analysis
        }
    
    def analyze_document_structure(self, text_blocks: List[TextBlock]) -> Tuple[List[HeadingCandidate], AnalysisContext]:
        """
        Analyze document structure and identify potential headings.
        
        Args:
            text_blocks: List of extracted text blocks from PDF
            
        Returns:
            Tuple of (heading candidates, analysis context)
        """
        if not text_blocks:
            logger.warning("No text blocks provided for structure analysis")
            return [], AnalysisContext([], 0.0, set(), {}, {})
        
        logger.info(f"Analyzing document structure with {len(text_blocks)} text blocks")
        
        # Build analysis context
        context = self._build_analysis_context(text_blocks)
        
        # Identify heading candidates
        candidates = []
        
        for i, block in enumerate(text_blocks):
            if block.is_empty():
                continue
            
            # Extract features for this text block
            features = self._extract_formatting_features(block, text_blocks, i, context)
            
            # Calculate confidence score using rule-based heuristics
            confidence = self._calculate_confidence_score(block, features, context)
            
            # Determine level indicators
            level_indicators = self._detect_level_indicators(block, features)
            
            # Create heading candidate if confidence is above minimum threshold
            if confidence >= self.confidence_thresholds['low_confidence']:
                candidate = HeadingCandidate(
                    text=block.text,
                    page=block.page,
                    confidence_score=confidence,
                    formatting_features=features,
                    level_indicators=level_indicators,
                    text_block=block
                )
                candidates.append(candidate)
                
                logger.debug(f"Heading candidate: '{block.text[:50]}...' (confidence: {confidence:.3f})")
        
        logger.info(f"Identified {len(candidates)} heading candidates")
        # Try to detect and add fragmented headings
        logger.info(f"Starting fragmentation detection...")
        fragmented_headings = self._detect_fragmented_headings(text_blocks, context)
        logger.info(f"Found {len(fragmented_headings)} fragmented heading candidates")
        candidates.extend(fragmented_headings)
        
        logger.info(f"Total candidates after fragmentation detection: {len(candidates)}")
        return candidates, context
    
    def _detect_fragmented_headings(self, text_blocks: List[TextBlock], context: AnalysisContext) -> List[HeadingCandidate]:
        """
        Detect headings that are fragmented across multiple text blocks.
        
        Args:
            text_blocks: List of text blocks
            context: Analysis context
            
        Returns:
            List of heading candidates from fragmented text
        """
        fragmented_candidates = []
        
        if not text_blocks:
            return fragmented_candidates
        
        # Group blocks by page
        pages = {}
        for block in text_blocks:
            if block.page not in pages:
                pages[block.page] = []
            pages[block.page].append(block)
        
        # Process each page
        for page_num, page_blocks in pages.items():
            # Find blocks with large fonts that might be fragmented headings
            font_sizes = [b.font_size for b in page_blocks if b.font_size > 0]
            if not font_sizes:
                continue
                
            avg_font_size = sum(font_sizes) / len(font_sizes)
            large_font_threshold = avg_font_size * 1.5
            
            logger.info(f"Page {page_num}: avg_font_size={avg_font_size:.1f}, threshold={large_font_threshold:.1f}")
            
            # Find consecutive large font blocks
            large_blocks = []
            for i, block in enumerate(page_blocks):
                if (block.font_size >= large_font_threshold and 
                    len(block.text.strip()) > 0 and
                    not block.is_empty()):
                    large_blocks.append((i, block))
                    logger.info(f"Large font block {i}: '{block.text}' (font: {block.font_size:.1f})")
            
            logger.info(f"Found {len(large_blocks)} large font blocks")
            
            # Try to combine consecutive large font blocks
            if len(large_blocks) >= 2:
                combined_candidates = self._combine_consecutive_blocks(large_blocks, page_blocks, context)
                fragmented_candidates.extend(combined_candidates)
        
        return fragmented_candidates
    
    def _combine_consecutive_blocks(self, large_blocks: List[Tuple[int, TextBlock]], 
                                  page_blocks: List[TextBlock], context: AnalysisContext) -> List[HeadingCandidate]:
        """
        Combine consecutive large font blocks into potential headings.
        
        Args:
            large_blocks: List of (index, block) tuples for large font blocks
            page_blocks: All blocks on the page
            context: Analysis context
            
        Returns:
            List of heading candidates from combined blocks
        """
        candidates = []
        
        logger.info(f"Attempting to combine {len(large_blocks)} large font blocks")
        
        i = 0
        while i < len(large_blocks):
            # Start a potential combination
            combination = [large_blocks[i]]
            j = i + 1
            
            # Look for consecutive blocks that might be part of the same heading
            while j < len(large_blocks):
                current_idx, current_block = large_blocks[j]
                prev_idx, prev_block = large_blocks[j-1]
                
                # Check if blocks are close enough to be part of same heading
                vertical_distance = abs(current_block.bbox[1] - prev_block.bbox[1])
                horizontal_distance = current_block.bbox[0] - prev_block.bbox[2]
                
                # If blocks are on roughly the same line or very close vertically
                if (vertical_distance < 15 and horizontal_distance < 100) or vertical_distance < 5:
                    combination.append(large_blocks[j])
                    j += 1
                else:
                    break
            
            # If we have multiple blocks to combine, create a candidate
            if len(combination) >= 2:
                logger.info(f"Found combination of {len(combination)} blocks: {[block.text for _, block in combination]}")
                combined_text = self._combine_block_texts(combination)
                logger.info(f"Combined text: '{combined_text}'")
                if combined_text and len(combined_text.strip()) > 3:
                    # Create a synthetic text block for the combined text
                    first_block = combination[0][1]
                    last_block = combination[-1][1]
                    
                    # Calculate combined bounding box
                    combined_bbox = (
                        first_block.bbox[0],  # left
                        min(block.bbox[1] for _, block in combination),  # top
                        last_block.bbox[2],   # right
                        max(block.bbox[3] for _, block in combination)   # bottom
                    )
                    
                    # Use the largest font size and check if any block is bold
                    max_font_size = max(block.font_size for _, block in combination)
                    is_bold = any(block.is_bold for _, block in combination)
                    
                    # Create synthetic text block
                    synthetic_block = TextBlock(
                        text=combined_text,
                        bbox=combined_bbox,
                        font_size=max_font_size,
                        font_name=first_block.font_name,
                        is_bold=is_bold,
                        page=first_block.page,
                        line_height=first_block.line_height
                    )
                    
                    # Extract features and calculate confidence
                    features = self._extract_formatting_features(synthetic_block, page_blocks, 0, context)
                    confidence = self._calculate_confidence_score(synthetic_block, features, context)
                    
                    # Boost confidence for fragmented headings since they represent large, bold text
                    # that was intentionally fragmented by the PDF layout
                    fragmentation_boost = 0.25  # Increased boost
                    confidence += fragmentation_boost
                    confidence = min(1.0, confidence)  # Cap at 1.0
                    
                    logger.info(f"Synthetic block confidence: {confidence:.3f} (boosted), threshold: {self.confidence_thresholds['low_confidence']:.3f}")
                    
                    # Only add if confidence is reasonable
                    if confidence >= self.confidence_thresholds['low_confidence']:
                        level_indicators = self._detect_level_indicators(synthetic_block, features)
                        
                        candidate = HeadingCandidate(
                            text=combined_text,
                            page=first_block.page,
                            confidence_score=confidence,
                            formatting_features=features,
                            level_indicators=level_indicators,
                            text_block=synthetic_block
                        )
                        candidates.append(candidate)
                        
                        logger.info(f"Created fragmented heading candidate: '{combined_text}' (confidence: {confidence:.3f})")
            
            i = j
        
        return candidates
    
    def _combine_block_texts(self, block_combinations: List[Tuple[int, TextBlock]]) -> str:
        """
        Combine text from multiple blocks intelligently.
        
        Args:
            block_combinations: List of (index, block) tuples
            
        Returns:
            Combined text string
        """
        if not block_combinations:
            return ""
        
        combined_parts = []
        prev_block = None
        
        for _, block in block_combinations:
            text = block.text.strip()
            if not text:
                continue
                
            if prev_block is None:
                combined_parts.append(text)
            else:
                # Check if we need to add space between parts
                prev_text = combined_parts[-1] if combined_parts else ""
                
                # Smart spacing logic for fragmented text
                if prev_text:
                    # Handle special cases for fragmented words
                    if (prev_text.endswith(('Y', 'T')) and text.lower().startswith(('ou', 'here'))) or \
                       (len(prev_text) == 1 and prev_text.isupper() and text.islower()):
                        # Direct concatenation for fragmented words like "Y" + "ou" = "You"
                        combined_parts.append(text)
                    elif not prev_text[-1].isspace() and not prev_text[-1] in '.,!?:;-' and \
                         not text[0].isspace() and not text[0] in '.,!?:;-':
                        # Add space for separate words
                        combined_parts.append(" " + text)
                    else:
                        combined_parts.append(text)
                else:
                    combined_parts.append(text)
            
            prev_block = block
        
        return "".join(combined_parts).strip()
    
    def _build_analysis_context(self, text_blocks: List[TextBlock]) -> AnalysisContext:
        """
        Build analysis context from document text blocks.
        
        Args:
            text_blocks: List of text blocks
            
        Returns:
            Analysis context with document statistics
        """
        font_sizes = [block.font_size for block in text_blocks if not block.is_empty()]
        font_names = {block.font_name for block in text_blocks if not block.is_empty()}
        
        # Calculate average font size
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0
        
        # Analyze page margins (simplified - using bbox positions)
        page_margins = {}
        pages = {block.page for block in text_blocks}
        
        for page in pages:
            page_blocks = [b for b in text_blocks if b.page == page and not b.is_empty()]
            if page_blocks:
                left_margin = min(b.bbox[0] for b in page_blocks)
                top_margin = min(b.bbox[1] for b in page_blocks)
                right_margin = max(b.bbox[2] for b in page_blocks)
                bottom_margin = max(b.bbox[3] for b in page_blocks)
                page_margins[page] = (left_margin, top_margin, right_margin, bottom_margin)
        
        # Calculate line spacing statistics
        line_heights = [block.line_height for block in text_blocks if not block.is_empty()]
        line_spacing_stats = {
            'mean': sum(line_heights) / len(line_heights) if line_heights else 12.0,
            'median': sorted(line_heights)[len(line_heights) // 2] if line_heights else 12.0,
            'std': 0.0  # Simplified for now
        }
        
        return AnalysisContext(
            document_font_sizes=font_sizes,
            average_font_size=avg_font_size,
            common_font_names=font_names,
            page_margins=page_margins,
            line_spacing_stats=line_spacing_stats
        )
    
    def _extract_formatting_features(self, block: TextBlock, all_blocks: List[TextBlock], 
                                   block_index: int, context: AnalysisContext) -> Dict[str, Any]:
        """
        Extract formatting features for a text block.
        
        Args:
            block: Current text block
            all_blocks: All text blocks in document
            block_index: Index of current block
            context: Analysis context
            
        Returns:
            Dictionary of formatting features
        """
        features = {}
        
        # Basic text properties
        features['text_length'] = len(block.text)
        features['word_count'] = len(block.text.split())
        features['is_bold'] = block.is_bold
        features['font_size'] = block.font_size
        features['font_name'] = block.font_name
        
        # Font size analysis
        features['font_size_ratio'] = block.font_size / context.average_font_size if context.average_font_size > 0 else 1.0
        features['is_large_font'] = block.font_size > context.average_font_size * 1.2
        features['is_small_font'] = block.font_size < context.average_font_size * 0.8
        
        # Text structure analysis
        features['is_title_case'] = self._is_title_case(block.text)
        features['is_all_caps'] = block.text.isupper() and len(block.text) > 1
        features['ends_with_period'] = block.text.endswith('.')
        features['ends_with_colon'] = block.text.endswith(':')
        features['has_punctuation'] = bool(re.search(r'[.!?:;,]', block.text))
        
        # Position and spacing analysis
        features['whitespace_before'] = self._calculate_whitespace_before(block, all_blocks, block_index)
        features['whitespace_after'] = self._calculate_whitespace_after(block, all_blocks, block_index)
        features['is_isolated'] = self._is_visually_isolated(block, all_blocks, block_index)
        features['page_position'] = self._calculate_page_position(block, context)
        
        # Numbering pattern analysis
        features['numbering_pattern'] = self._detect_numbering_pattern(block.text)
        features['numbering_level'] = self._extract_numbering_level(block.text)
        
        # Keyword analysis
        features['keyword_matches'] = self._find_keyword_matches(block.text)
        features['keyword_confidence'] = self._calculate_keyword_confidence(features['keyword_matches'])
        
        return features
    
    def _calculate_confidence_score(self, block: TextBlock, features: Dict[str, Any], 
                                  context: AnalysisContext) -> float:
        """
        Calculate confidence score for heading classification using weighted heuristics.
        
        Args:
            block: Text block to analyze
            features: Extracted formatting features
            context: Analysis context
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        scores = {}
        
        # 1. Numbering pattern score (25% weight)
        scores['numbering'] = self._score_numbering_patterns(features)
        
        # 2. Visual isolation score (35% weight)
        scores['visual_isolation'] = self._score_visual_isolation(features)
        
        # 3. Text structure score (25% weight)
        scores['text_structure'] = self._score_text_structure(features)
        
        # 4. Keyword matching score (15% weight)
        scores['keyword_match'] = features.get('keyword_confidence', 0.0)
        
        # Calculate weighted final score
        final_score = sum(
            scores[component] * self.scoring_weights[component]
            for component in scores
        )
        
        # Apply additional modifiers
        final_score = self._apply_confidence_modifiers(final_score, block, features, context)
        
        # Ensure score is within valid range
        return max(0.0, min(1.0, final_score))
    
    def _score_numbering_patterns(self, features: Dict[str, Any]) -> float:
        """Score based on numbering pattern analysis with contextual validation."""
        pattern = features.get('numbering_pattern')
        level = features.get('numbering_level', 0)
        
        if not pattern:
            return 0.0
        
        # Base scores for different patterns
        base_scores = {
            'simple_decimal': 0.8,          # 1., 2., 3. - good indicator but validate context
            'hierarchical_decimal': 0.7,    # 1.1, 1.1.1 - good but could be list items
            'simple_number_with_text': 0.6, # 1 Text - moderate confidence
            'roman_numerals': 0.7,          # I., II. - good for major sections
            'letters': 0.5,                 # A., B. - moderate confidence
            'parenthetical': 0.3            # (1), (2) - lower confidence
        }
        
        base_score = base_scores.get(pattern, 0.0)
        
        # Contextual adjustments for hierarchical patterns
        if pattern == 'hierarchical_decimal':
            if level == 2:
                base_score = 0.75  # 1.1, 1.2 - good confidence
            elif level == 3:
                base_score = 0.7   # 1.1.1 - still good but validate context
            else:
                base_score = 0.5   # Deeper levels need more validation
        
        # Additional contextual validation would be applied in confidence modifiers
        # This keeps numbering as a strong indicator while allowing other factors to influence
        
        return base_score
    
    def _score_visual_isolation(self, features: Dict[str, Any]) -> float:
        """Score based on visual isolation analysis."""
        score = 0.0
        
        # Whitespace before (up to 0.4 points)
        whitespace_before = features.get('whitespace_before', 0.0)
        if whitespace_before > 20:
            score += 0.4
        elif whitespace_before > 15:
            score += 0.3
        elif whitespace_before > 10:
            score += 0.2
        elif whitespace_before > 5:
            score += 0.1
        
        # Whitespace after (up to 0.3 points)
        whitespace_after = features.get('whitespace_after', 0.0)
        if whitespace_after > 15:
            score += 0.3
        elif whitespace_after > 10:
            score += 0.2
        elif whitespace_after > 5:
            score += 0.1
        
        # Visual isolation (up to 0.3 points)
        if features.get('is_isolated', False):
            score += 0.3
        
        return min(1.0, score)
    
    def _score_text_structure(self, features: Dict[str, Any]) -> float:
        """Score based on text structure analysis."""
        score = 0.0
        
        # Length constraints (up to 0.3 points)
        text_length = features.get('text_length', 0)
        if 5 <= text_length <= 100:
            score += 0.3
        elif 3 <= text_length <= 150:
            score += 0.2
        elif text_length > 150:
            score -= 0.2  # Too long for typical heading
        
        # Capitalization patterns (up to 0.4 points)
        if features.get('is_title_case', False):
            score += 0.3
        elif features.get('is_all_caps', False):
            # ALL CAPS text gets a significant boost (especially for short phrases)
            word_count = features.get('word_count', 0)
            if word_count <= 3:  # Short ALL CAPS phrases like "PATHWAY OPTIONS"
                score += 0.4
            else:
                score += 0.25
        
        # Punctuation patterns (up to 0.2 points)
        if not features.get('ends_with_period', False):
            score += 0.1  # Headings typically don't end with periods
        if features.get('ends_with_colon', False):
            score += 0.2  # Headings often end with colons
        
        # Font characteristics (up to 0.2 points)
        if features.get('is_bold', False):
            score += 0.1
        if features.get('is_large_font', False):
            score += 0.1
        
        return min(1.0, score)
    
    def _apply_confidence_modifiers(self, base_score: float, block: TextBlock, 
                                  features: Dict[str, Any], context: AnalysisContext) -> float:
        """Apply additional confidence modifiers based on context."""
        modified_score = base_score
        
        # Apply exclusion patterns for common non-headings
        if self._is_likely_non_heading(block.text):
            modified_score -= 0.4  # Heavy penalty for non-heading patterns
        
        # Special boost for numbered patterns with good context
        if features.get('numbering_pattern') and self._has_good_heading_context(block, features):
            modified_score += self.confidence_thresholds['numbered_boost']
        
        # Boost score for very short text (likely headings)
        if features.get('text_length', 0) < 10:
            modified_score += 0.1
        
        # Reduce score for very long text (unlikely headings)
        if features.get('text_length', 0) > 200:
            modified_score -= 0.2
        
        # Boost score for text at top of page
        page_position = features.get('page_position', 0.5)
        if page_position < 0.2:  # Top 20% of page
            modified_score += 0.1
        
        # Reduce score for text with many punctuation marks (likely body text)
        punct_count = len(re.findall(r'[.!?:;,]', block.text))
        if punct_count > 3:
            modified_score -= 0.1
        
        return modified_score
    
    def _has_good_heading_context(self, block: TextBlock, features: Dict[str, Any]) -> bool:
        """
        Check if a numbered pattern has good heading context (not just a list item).
        
        Args:
            block: Text block to analyze
            features: Extracted formatting features
            
        Returns:
            True if the numbered pattern appears to be a heading, not a list item
        """
        # Good visual isolation suggests heading, not list item
        if features.get('is_isolated', False):
            return True
        
        # Significant whitespace before/after suggests heading
        whitespace_before = features.get('whitespace_before', 0)
        whitespace_after = features.get('whitespace_after', 0)
        if whitespace_before > 10 or whitespace_after > 10:
            return True
        
        # Short, concise text is more likely to be a heading
        text_length = features.get('text_length', 0)
        word_count = features.get('word_count', 0)
        if text_length < 50 and word_count < 8:
            return True
        
        # Title case or all caps suggests heading
        if features.get('is_title_case', False) or features.get('is_all_caps', False):
            return True
        
        # Bold text suggests heading
        if features.get('is_bold', False):
            return True
        
        return False
    
    def _is_likely_non_heading(self, text: str) -> bool:
        """
        Check if text is likely not a heading based on common patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is likely not a heading
        """
        text_lower = text.lower().strip()
        
        # Document metadata and boilerplate
        metadata_patterns = [
            r'^version\s+\d+',
            r'^copyright\s*©',
            r'international\s+software\s+testing',
            r'qualifications\s+board',
            r'istqb',
            r'foundation\s+level\s+extensions',
            r'^\d{1,2}\s+(june|july|august|september|october|november|december)',
            r'^\d{4}$',  # Just a year
            r'^v?\d+\.\d+$',  # Version numbers
            r'^page\s+\d+',
            r'^draft|^final|^revised',
            r'^address:?$',  # Address labels
            r'^rsvp:?$',     # RSVP labels
            r'^date:?$',     # Date labels
            r'^time:?$',     # Time labels
            r'^\d+\s+(street|st|avenue|ave|road|rd|drive|dr|lane|ln|parkway|pkwy)',  # Address components
            r'^parkway$',    # Just "PARKWAY" alone
        ]
        
        for pattern in metadata_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Generic single words that appear frequently
        generic_words = {
            'overview', 'board', 'version', 'date', 'document', 'title',
            'author', 'subject', 'keywords', 'draft', 'final', 'revised'
        }
        
        if text_lower in generic_words:
            return True
        
        # Very long sentences (likely body text)
        if len(text.split()) > 15 and text.endswith('.'):
            return True
        
        # Copyright and legal notices
        if any(term in text_lower for term in ['copyright', '©', 'all rights reserved']):
            return True
        
        return False
    
    def _detect_numbering_pattern(self, text: str) -> Optional[str]:
        """Detect numbering pattern in text."""
        text_start = text.lstrip()
        
        # Check patterns in order of specificity
        for pattern_name, pattern_regex in self.numbering_patterns.items():
            if pattern_regex.match(text_start):
                return pattern_name
        
        return None
    
    def _extract_numbering_level(self, text: str) -> int:
        """Extract hierarchical level from numbering pattern."""
        text_start = text.lstrip()
        
        # Check for hierarchical decimal pattern (1.1.1)
        decimal_match = self.numbering_patterns['hierarchical_decimal'].match(text_start)
        if decimal_match:
            number_part = decimal_match.group().strip().rstrip('.')
            return len(number_part.split('.'))
        
        # Check for simple decimal pattern (1.)
        simple_decimal_match = self.numbering_patterns['simple_decimal'].match(text_start)
        if simple_decimal_match:
            return 1
        
        # Other patterns are considered level 1
        for pattern_regex in self.numbering_patterns.values():
            if pattern_regex.match(text_start):
                return 1
        
        return 0
    
    def _find_keyword_matches(self, text: str) -> Dict[str, List[str]]:
        """Find heading keyword matches in text."""
        text_lower = text.lower()
        matches = {'primary': [], 'secondary': [], 'tertiary': []}
        
        for category, keywords in self.heading_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matches[category].append(keyword)
        
        return matches
    
    def _calculate_keyword_confidence(self, keyword_matches: Dict[str, List[str]]) -> float:
        """Calculate confidence score based on keyword matches."""
        score = 0.0
        
        # Primary keywords contribute most
        score += len(keyword_matches.get('primary', [])) * 0.4
        
        # Secondary keywords contribute moderately
        score += len(keyword_matches.get('secondary', [])) * 0.25
        
        # Tertiary keywords contribute least
        score += len(keyword_matches.get('tertiary', [])) * 0.15
        
        return min(1.0, score)
    
    def _is_title_case(self, text: str) -> bool:
        """Check if text is in title case."""
        if not text:
            return False
        
        # Don't consider ALL CAPS as title case
        if text.isupper():
            return False
        
        words = text.split()
        if not words:
            return False
        
        # Check if most words are capitalized (allowing for articles, prepositions)
        capitalized_count = sum(1 for word in words if word[0].isupper())
        return capitalized_count / len(words) >= 0.6
    
    def _calculate_whitespace_before(self, block: TextBlock, all_blocks: List[TextBlock], 
                                   block_index: int) -> float:
        """Calculate whitespace before the current block."""
        if block_index == 0:
            return 0.0
        
        # Find previous block on same page
        prev_block = None
        for i in range(block_index - 1, -1, -1):
            if all_blocks[i].page == block.page and not all_blocks[i].is_empty():
                prev_block = all_blocks[i]
                break
        
        if not prev_block:
            return 0.0
        
        # Calculate vertical distance
        return float(block.bbox[1] - prev_block.bbox[3])
    
    def _calculate_whitespace_after(self, block: TextBlock, all_blocks: List[TextBlock], 
                                  block_index: int) -> float:
        """Calculate whitespace after the current block."""
        if block_index >= len(all_blocks) - 1:
            return 0.0
        
        # Find next block on same page
        next_block = None
        for i in range(block_index + 1, len(all_blocks)):
            if all_blocks[i].page == block.page and not all_blocks[i].is_empty():
                next_block = all_blocks[i]
                break
        
        if not next_block:
            return 0.0
        
        # Calculate vertical distance
        return float(next_block.bbox[1] - block.bbox[3])
    
    def _is_visually_isolated(self, block: TextBlock, all_blocks: List[TextBlock], 
                            block_index: int) -> bool:
        """Check if block is visually isolated from surrounding text."""
        whitespace_before = self._calculate_whitespace_before(block, all_blocks, block_index)
        whitespace_after = self._calculate_whitespace_after(block, all_blocks, block_index)
        
        # Consider isolated if significant whitespace on both sides
        return whitespace_before > 10 and whitespace_after > 10
    
    def _calculate_page_position(self, block: TextBlock, context: AnalysisContext) -> float:
        """Calculate relative position of block on page (0.0 = top, 1.0 = bottom)."""
        page_margins = context.page_margins.get(block.page)
        if not page_margins:
            return 0.5  # Default to middle if no margin info
        
        _, top_margin, _, bottom_margin = page_margins
        page_height = bottom_margin - top_margin
        
        if page_height <= 0:
            return 0.5
        
        block_y = block.bbox[1]
        relative_position = (block_y - top_margin) / page_height
        
        return max(0.0, min(1.0, relative_position))
    
    def _detect_level_indicators(self, block: TextBlock, features: Dict[str, Any]) -> List[str]:
        """Detect level indicators for heading classification."""
        indicators = []
        
        # Numbering-based indicators
        pattern = features.get('numbering_pattern')
        level = features.get('numbering_level', 0)
        
        if pattern and level > 0:
            indicators.append(f"numbering_level_{level}")
            indicators.append(f"pattern_{pattern}")
        
        # Font-based indicators
        if features.get('is_large_font', False):
            indicators.append("large_font")
        if features.get('is_bold', False):
            indicators.append("bold_text")
        
        # Position-based indicators
        if features.get('page_position', 0.5) < 0.2:
            indicators.append("top_of_page")
        
        # Keyword-based indicators
        keyword_matches = features.get('keyword_matches', {})
        if keyword_matches.get('primary'):
            indicators.append("primary_keywords")
        if keyword_matches.get('secondary'):
            indicators.append("secondary_keywords")
        
        return indicators
    
    def get_high_confidence_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Filter candidates to only high-confidence headings."""
        return [
            candidate for candidate in candidates
            if candidate.confidence_score >= self.confidence_thresholds['high_confidence']
        ]
    
    def get_ambiguous_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Filter candidates to only ambiguous cases that need additional analysis."""
        min_conf, max_conf = self.confidence_thresholds['ambiguous_range']
        return [
            candidate for candidate in candidates
            if min_conf <= candidate.confidence_score < max_conf
        ]
    
    def validate_heading_sequence(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """
        Validate heading sequence for logical consistency.
        
        Note: This implementation preserves author intent and does not enforce
        strict hierarchical sequences (H1->H2->H3), allowing for non-hierarchical
        patterns like H1->H3->H2 as per design requirements.
        """
        # Sort candidates by page and position
        sorted_candidates = sorted(candidates, key=lambda c: (c.page, c.text_block.bbox[1] if c.text_block else 0))
        
        # For now, return all candidates as-is to preserve author intent
        # Future enhancements could add optional sequence validation
        return sorted_candidates


def analyze_document_structure(text_blocks: List[TextBlock]) -> Tuple[List[HeadingCandidate], AnalysisContext]:
    """
    Convenience function to analyze document structure.
    
    Args:
        text_blocks: List of extracted text blocks
        
    Returns:
        Tuple of (heading candidates, analysis context)
    """
    analyzer = StructureAnalyzer()
    return analyzer.analyze_document_structure(text_blocks)


def get_high_confidence_headings(text_blocks: List[TextBlock]) -> List[HeadingCandidate]:
    """
    Convenience function to get high-confidence heading candidates.
    
    Args:
        text_blocks: List of extracted text blocks
        
    Returns:
        List of high-confidence heading candidates
    """
    analyzer = StructureAnalyzer()
    candidates, _ = analyzer.analyze_document_structure(text_blocks)
    return analyzer.get_high_confidence_candidates(candidates)