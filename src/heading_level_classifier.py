"""
Heading level classifier module for PDF outline extraction.

This module provides functionality to classify heading candidates into H1, H2, H3 levels
using multiple strategies including numbering hierarchy analysis and font size relationships.
Supports non-hierarchical sequences and handles edge cases as per design requirements.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter

from .data_models import HeadingCandidate, TextBlock
from .logging_config import setup_logging

logger = setup_logging()


@dataclass
class LevelAssignmentContext:
    """Context information for heading level assignment."""
    document_font_sizes: List[float]
    average_font_size: float
    font_size_clusters: Dict[float, int]  # font_size -> cluster_id
    numbering_sequences: Dict[str, List[HeadingCandidate]]  # pattern -> candidates
    page_order: List[HeadingCandidate]  # candidates in document order


class HeadingLevelClassifier:
    """
    Classifies heading candidates into H1, H2, H3 levels using multi-factor analysis.
    
    Implements level assignment based on:
    - Primary: Numbering hierarchy analysis (1.=H1, 1.1=H2, 1.1.1=H3)
    - Secondary: Font size relationship analysis
    - Tertiary: Semantic importance and visual prominence
    
    Supports non-hierarchical sequences (H1→H3→H2) without correction,
    preserving author intent and document structure.
    """
    
    def __init__(self):
        """Initialize the heading level classifier."""
        # Numbering pattern analysis (order matters - more specific first)
        self.numbering_patterns = [
            ('hierarchical_decimal', re.compile(r'^(\d+\.\d+(?:\.\d+)*)\s*(.*)$')),  # 1.1 or 1.1.1 (multi-level)
            ('simple_decimal', re.compile(r'^(\d+)\.\s*(.*)$')),  # 1. Text (single level)
            ('simple_number_with_text', re.compile(r'^(\d+)\s+(.*)$')),  # 1 Text
            ('roman_numerals', re.compile(r'^([IVX]+)\.\s*(.*)$', re.IGNORECASE)),  # I. Text
            ('letters', re.compile(r'^([A-Z])\.\s*(.*)$')),  # A. Text
            ('parenthetical', re.compile(r'^\((\d+)\)\s*(.*)$')),  # (1) Text
        ]
        
        # Level assignment weights for multi-factor scoring
        self.level_weights = {
            'numbering_hierarchy': 0.8,    # Primary indicator (increased for better numbered heading detection)
            'font_size_analysis': 0.15,    # Secondary indicator
            'semantic_importance': 0.04,   # Tertiary indicator
            'visual_prominence': 0.01      # Supporting indicator
        }
        
        # Font size clustering tolerance
        self.font_size_tolerance = 0.5  # Points tolerance for clustering
        
    def classify_heading_levels(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """
        Classify heading candidates into H1, H2, H3 levels.
        
        Args:
            candidates: List of heading candidates to classify
            
        Returns:
            List of candidates with assigned heading levels
        """
        if not candidates:
            logger.warning("No heading candidates provided for level classification")
            return []
        
        logger.info(f"Classifying heading levels for {len(candidates)} candidates")
        
        # Build classification context
        context = self._build_classification_context(candidates)
        
        # Classify each candidate independently
        classified_candidates = []
        
        for candidate in candidates:
            # Determine heading level using multi-factor analysis
            level = self._determine_heading_level(candidate, context)
            
            # Create new candidate with assigned level
            classified_candidate = self._create_classified_candidate(candidate, level)
            classified_candidates.append(classified_candidate)
            
            logger.debug(f"Classified '{candidate.text[:50]}...' as H{level}")
        
        # Sort by document order (page, then position)
        classified_candidates.sort(key=lambda c: (c.page, c.text_block.bbox[1] if c.text_block else 0))
        
        logger.info(f"Level classification complete: {self._get_level_distribution(classified_candidates)}")
        return classified_candidates
    
    def _build_classification_context(self, candidates: List[HeadingCandidate]) -> LevelAssignmentContext:
        """
        Build context for heading level classification.
        
        Args:
            candidates: List of heading candidates
            
        Returns:
            Classification context with analysis data
        """
        # Extract font sizes from candidates
        font_sizes = []
        for candidate in candidates:
            if candidate.text_block:
                font_sizes.append(candidate.text_block.font_size)
        
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0
        
        # Cluster font sizes
        font_clusters = self._cluster_font_sizes(font_sizes)
        
        # Group candidates by numbering patterns
        numbering_sequences = self._group_by_numbering_patterns(candidates)
        
        # Sort candidates by document order
        page_order = sorted(candidates, key=lambda c: (c.page, c.text_block.bbox[1] if c.text_block else 0))
        
        return LevelAssignmentContext(
            document_font_sizes=font_sizes,
            average_font_size=avg_font_size,
            font_size_clusters=font_clusters,
            numbering_sequences=numbering_sequences,
            page_order=page_order
        )
    
    def _determine_heading_level(self, candidate: HeadingCandidate, context: LevelAssignmentContext) -> int:
        """
        Determine heading level for a candidate using multi-factor analysis.
        
        Args:
            candidate: Heading candidate to classify
            context: Classification context
            
        Returns:
            Heading level (1, 2, or 3)
        """
        # Calculate level scores for each factor
        scores = {}
        
        # 1. Numbering hierarchy analysis (primary, 60% weight)
        scores['numbering'] = self._analyze_numbering_hierarchy(candidate, context)
        
        # 2. Font size relationship analysis (secondary, 25% weight)
        scores['font_size'] = self._analyze_font_size_relationships(candidate, context)
        
        # 3. Semantic importance analysis (tertiary, 10% weight)
        scores['semantic'] = self._analyze_semantic_importance(candidate, context)
        
        # 4. Visual prominence analysis (supporting, 5% weight)
        scores['visual'] = self._analyze_visual_prominence(candidate, context)
        
        # Determine level based on weighted scores
        level = self._calculate_final_level(scores, candidate, context)
        
        logger.debug(f"Level scores for '{candidate.text[:30]}...': {scores} -> H{level}")
        return level
    
    def _analyze_numbering_hierarchy(self, candidate: HeadingCandidate, context: LevelAssignmentContext) -> Dict[int, float]:
        """
        Analyze numbering hierarchy to determine level preferences.
        
        Args:
            candidate: Heading candidate
            context: Classification context
            
        Returns:
            Dictionary mapping level (1,2,3) to confidence score
        """
        scores = {1: 0.0, 2: 0.0, 3: 0.0}
        
        # Extract numbering information from text
        numbering_info = self._extract_numbering_info(candidate.text)
        
        if not numbering_info:
            # No numbering pattern - return neutral scores
            return scores
        
        pattern_type, number_parts, _ = numbering_info
        
        # Hierarchical decimal patterns (1.1, 1.1.1)
        if pattern_type == 'hierarchical_decimal':
            level = len(number_parts)
            if level == 2:
                scores[2] = 1.0  # 1.1 -> H2
            elif level >= 3:
                scores[3] = 1.0  # 1.1.1 -> H3
            else:
                scores[1] = 0.8  # Fallback
        
        # Simple decimal patterns (1.)
        elif pattern_type == 'simple_decimal':
            scores[1] = 1.0  # Strong H1 indicator
        
        # Simple number patterns with text (1 Text)
        elif pattern_type == 'simple_number_with_text':
            scores[1] = 0.8  # Moderate H1 indicator
        
        # Roman numerals (I.)
        elif pattern_type == 'roman_numerals':
            scores[1] = 0.9  # Strong H1 indicator
        
        # Letters (A.)
        elif pattern_type == 'letters':
            scores[2] = 0.8  # Moderate H2 indicator
        
        # Parenthetical numbering ((1))
        elif pattern_type == 'parenthetical':
            scores[2] = 0.6  # Weak H2 indicator
        
        return scores
    
    def _analyze_font_size_relationships(self, candidate: HeadingCandidate, context: LevelAssignmentContext) -> Dict[int, float]:
        """
        Analyze font size relationships to determine level preferences.
        
        Args:
            candidate: Heading candidate
            context: Classification context
            
        Returns:
            Dictionary mapping level (1,2,3) to confidence score
        """
        scores = {1: 0.0, 2: 0.0, 3: 0.0}
        
        if not candidate.text_block:
            return scores
        
        # Use pre-calculated font size ratio from structure analyzer (more accurate)
        size_ratio = candidate.formatting_features.get('font_size_ratio', 1.0)
        
        # Determine level preferences based on font size
        if size_ratio >= 1.4:
            # Very large font - likely H1
            scores[1] = 1.0
            scores[2] = 0.3
            scores[3] = 0.1
        elif size_ratio >= 1.2:
            # Large font - likely H1 or H2
            scores[1] = 0.8
            scores[2] = 0.6
            scores[3] = 0.2
        elif size_ratio >= 1.1:
            # Slightly large font - could be any level
            scores[1] = 0.6
            scores[2] = 0.8
            scores[3] = 0.4
        elif size_ratio >= 0.9:
            # Normal font size - likely H2 or H3
            scores[1] = 0.4
            scores[2] = 0.7
            scores[3] = 0.8
        else:
            # Small font - likely H3
            scores[1] = 0.2
            scores[2] = 0.5
            scores[3] = 1.0
        
        # Boost scores for bold text
        if candidate.text_block.is_bold:
            scores[1] += 0.3
            scores[2] += 0.2
            scores[3] += 0.1
        
        # Normalize scores to max 1.0
        for level in scores:
            scores[level] = min(1.0, scores[level])
        
        return scores
    
    def _analyze_semantic_importance(self, candidate: HeadingCandidate, context: LevelAssignmentContext) -> Dict[int, float]:
        """
        Analyze semantic importance to determine level preferences.
        
        Args:
            candidate: Heading candidate
            context: Classification context
            
        Returns:
            Dictionary mapping level (1,2,3) to confidence score
        """
        scores = {1: 0.0, 2: 0.0, 3: 0.0}
        
        text_lower = candidate.text.lower()
        
        # High-level semantic indicators (H1)
        h1_keywords = {
            'introduction', 'conclusion', 'abstract', 'summary', 'overview',
            'chapter', 'part', 'executive summary', 'background', 'methodology',
            'results', 'discussion', 'references', 'bibliography', 'appendix'
        }
        
        # Mid-level semantic indicators (H2)
        h2_keywords = {
            'section', 'analysis', 'evaluation', 'assessment', 'review',
            'findings', 'recommendations', 'implications', 'approach',
            'method', 'procedure', 'process', 'implementation'
        }
        
        # Low-level semantic indicators (H3)
        h3_keywords = {
            'definition', 'example', 'case study', 'problem', 'solution',
            'technique', 'tool', 'feature', 'component', 'detail',
            'specification', 'requirement', 'constraint'
        }
        
        # Check for keyword matches
        for keyword in h1_keywords:
            if keyword in text_lower:
                scores[1] += 0.6
        
        for keyword in h2_keywords:
            if keyword in text_lower:
                scores[2] += 0.5
        
        for keyword in h3_keywords:
            if keyword in text_lower:
                scores[3] += 0.4
        
        # Text length analysis
        text_length = len(candidate.text)
        if text_length < 20:
            # Short text - could be any level, slight preference for higher levels
            scores[1] += 0.2
            scores[2] += 0.1
            scores[3] += 0.1
        elif text_length < 50:
            # Medium text - typical heading length
            scores[1] += 0.3
            scores[2] += 0.3
            scores[3] += 0.2
        else:
            # Long text - less likely to be high-level heading
            scores[1] += 0.1
            scores[2] += 0.2
            scores[3] += 0.3
        
        # Normalize scores
        for level in scores:
            scores[level] = min(1.0, scores[level])
        
        return scores
    
    def _analyze_visual_prominence(self, candidate: HeadingCandidate, context: LevelAssignmentContext) -> Dict[int, float]:
        """
        Analyze visual prominence to determine level preferences.
        
        Args:
            candidate: Heading candidate
            context: Classification context
            
        Returns:
            Dictionary mapping level (1,2,3) to confidence score
        """
        scores = {1: 0.0, 2: 0.0, 3: 0.0}
        
        # Page position analysis
        if 'page_position' in candidate.formatting_features:
            page_pos = candidate.formatting_features['page_position']
            if page_pos < 0.2:  # Top 20% of page
                scores[1] += 0.5
                scores[2] += 0.2
        
        # Visual isolation analysis
        if candidate.formatting_features.get('is_isolated', False):
            scores[1] += 0.3
            scores[2] += 0.2
        
        # Whitespace analysis
        whitespace_before = candidate.formatting_features.get('whitespace_before', 0)
        whitespace_after = candidate.formatting_features.get('whitespace_after', 0)
        
        if whitespace_before > 20:
            scores[1] += 0.3
            scores[2] += 0.2
        
        if whitespace_after > 15:
            scores[1] += 0.2
            scores[2] += 0.1
        
        # Normalize scores
        for level in scores:
            scores[level] = min(1.0, scores[level])
        
        return scores
    
    def _calculate_final_level(self, scores: Dict[str, Dict[int, float]], 
                             candidate: HeadingCandidate, context: LevelAssignmentContext) -> int:
        """
        Calculate final heading level based on weighted factor scores.
        
        Args:
            scores: Dictionary of factor scores
            candidate: Heading candidate
            context: Classification context
            
        Returns:
            Final heading level (1, 2, or 3)
        """
        # Calculate weighted scores for each level
        final_scores = {1: 0.0, 2: 0.0, 3: 0.0}
        
        # Map score keys to weight keys
        score_to_weight = {
            'numbering': 'numbering_hierarchy',
            'font_size': 'font_size_analysis', 
            'semantic': 'semantic_importance',
            'visual': 'visual_prominence'
        }
        
        # Check if numbering analysis found any patterns
        has_numbering = any(scores.get('numbering', {}).values())
        
        # Adjust weights if no numbering pattern is found
        if not has_numbering:
            # Redistribute numbering weight to font size and visual analysis
            adjusted_weights = {
                'numbering_hierarchy': 0.0,     # No numbering pattern
                'font_size_analysis': 0.6,     # Promote font size to primary
                'semantic_importance': 0.2,    # Increase semantic importance
                'visual_prominence': 0.2       # Increase visual prominence
            }
        else:
            adjusted_weights = self.level_weights
        
        for factor, factor_scores in scores.items():
            weight_key = score_to_weight.get(factor, factor)
            weight = adjusted_weights.get(weight_key, 0.0)
            for level, score in factor_scores.items():
                final_scores[level] += score * weight
        
        # Find the level with highest score
        best_level = max(final_scores.keys(), key=lambda k: final_scores[k])
        max_score = final_scores[best_level]
        
        # Apply confidence threshold - if no clear winner, default to H2
        if max_score < 0.2:  # Very low confidence in any level
            logger.debug(f"Very low confidence in level assignment for '{candidate.text[:30]}...', defaulting to H2")
            return 2
        
        # Check for ties - prefer lower level numbers in case of ties
        tied_levels = [level for level, score in final_scores.items() if abs(score - max_score) < 0.05]
        if len(tied_levels) > 1:
            return min(tied_levels)
        
        return best_level
    
    def _extract_numbering_info(self, text: str) -> Optional[Tuple[str, List[str], str]]:
        """
        Extract numbering information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (pattern_type, number_parts, remaining_text) or None
        """
        text_stripped = text.strip()
        
        for pattern_name, pattern_regex in self.numbering_patterns:
            match = pattern_regex.match(text_stripped)
            if match:
                if pattern_name == 'hierarchical_decimal':
                    number_part = match.group(1)
                    remaining_text = match.group(2)
                    number_parts = number_part.split('.')
                    return (pattern_name, number_parts, remaining_text)
                else:
                    number_part = match.group(1)
                    remaining_text = match.group(2)
                    return (pattern_name, [number_part], remaining_text)
        
        return None    

    def _cluster_font_sizes(self, font_sizes: List[float]) -> Dict[float, int]:
        """
        Cluster font sizes to identify distinct size groups.
        
        Args:
            font_sizes: List of font sizes
            
        Returns:
            Dictionary mapping font_size to cluster_id
        """
        if not font_sizes:
            return {}
        
        # Sort unique font sizes
        unique_sizes = sorted(set(font_sizes))
        
        # Simple clustering based on tolerance
        clusters = {}
        cluster_id = 0
        
        for size in unique_sizes:
            # Check if this size belongs to an existing cluster
            assigned = False
            for existing_size, existing_cluster in clusters.items():
                if abs(size - existing_size) <= self.font_size_tolerance:
                    clusters[size] = existing_cluster
                    assigned = True
                    break
            
            # Create new cluster if not assigned
            if not assigned:
                clusters[size] = cluster_id
                cluster_id += 1
        
        return clusters
    
    def _group_by_numbering_patterns(self, candidates: List[HeadingCandidate]) -> Dict[str, List[HeadingCandidate]]:
        """
        Group candidates by their numbering patterns.
        
        Args:
            candidates: List of heading candidates
            
        Returns:
            Dictionary mapping pattern_type to list of candidates
        """
        groups = defaultdict(list)
        
        for candidate in candidates:
            numbering_info = self._extract_numbering_info(candidate.text)
            if numbering_info:
                pattern_type, _, _ = numbering_info
                groups[pattern_type].append(candidate)
            else:
                groups['no_numbering'].append(candidate)
        
        return dict(groups)
    
    def _create_classified_candidate(self, candidate: HeadingCandidate, level: int) -> HeadingCandidate:
        """
        Create a new candidate with assigned heading level.
        
        Args:
            candidate: Original candidate
            level: Assigned heading level
            
        Returns:
            New candidate with level information
        """
        # Copy original candidate
        new_candidate = HeadingCandidate(
            text=candidate.text,
            page=candidate.page,
            confidence_score=candidate.confidence_score,
            formatting_features=candidate.formatting_features.copy(),
            level_indicators=candidate.level_indicators.copy(),
            text_block=candidate.text_block,
            assigned_level=level
        )
        
        # Add level information
        new_candidate.formatting_features['assigned_level'] = level
        new_candidate.add_level_indicator(f"heading_level_{level}")
        
        return new_candidate
    
    def _get_level_distribution(self, candidates: List[HeadingCandidate]) -> Dict[int, int]:
        """
        Get distribution of heading levels in classified candidates.
        
        Args:
            candidates: List of classified candidates
            
        Returns:
            Dictionary mapping level to count
        """
        distribution = Counter()
        
        for candidate in candidates:
            level = candidate.formatting_features.get('assigned_level', 0)
            if level in [1, 2, 3]:
                distribution[level] += 1
        
        return dict(distribution)
    
    def validate_level_assignments(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """
        Validate level assignments for consistency and handle edge cases.
        
        Note: This method preserves non-hierarchical sequences (H1→H3→H2) as per
        design requirements, focusing on individual heading characteristics rather
        than enforcing strict hierarchical flow.
        
        Args:
            candidates: List of classified candidates
            
        Returns:
            List of candidates with validated level assignments
        """
        if not candidates:
            return []
        
        logger.info("Validating level assignments (preserving non-hierarchical sequences)")
        
        # Sort candidates by document order
        sorted_candidates = sorted(candidates, key=lambda c: (c.page, c.text_block.bbox[1] if c.text_block else 0))
        
        # Log the sequence for debugging
        sequence = [c.formatting_features.get('assigned_level', 0) for c in sorted_candidates]
        logger.debug(f"Heading level sequence: {sequence}")
        
        # Check for edge cases but don't modify levels (preserve author intent)
        self._log_edge_cases(sorted_candidates)
        
        return sorted_candidates
    
    def _log_edge_cases(self, candidates: List[HeadingCandidate]) -> None:
        """
        Log edge cases in heading sequences for debugging purposes.
        
        Args:
            candidates: List of classified candidates in document order
        """
        if len(candidates) < 2:
            return
        
        levels = [c.formatting_features.get('assigned_level', 0) for c in candidates]
        
        # Check for skipped levels (H1 → H3)
        for i in range(len(levels) - 1):
            current_level = levels[i]
            next_level = levels[i + 1]
            
            if next_level - current_level > 1:
                logger.debug(f"Skipped level detected: H{current_level} → H{next_level} "
                           f"('{candidates[i].text[:30]}...' → '{candidates[i+1].text[:30]}...')")
        
        # Check for orphaned sub-headings (H3 without H2 parent)
        for i, level in enumerate(levels):
            if level == 3:
                # Look for H2 parent in previous headings
                has_h2_parent = any(l == 2 for l in levels[:i])
                if not has_h2_parent:
                    logger.debug(f"Orphaned H3 detected: '{candidates[i].text[:30]}...' "
                               f"(no H2 parent found)")
        
        # Check for repeated levels
        level_counts = Counter(levels)
        for level, count in level_counts.items():
            if count > 1 and level in [1, 2, 3]:
                logger.debug(f"Repeated H{level} level detected: {count} instances")


def classify_heading_levels(candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
    """
    Convenience function to classify heading levels.
    
    Args:
        candidates: List of heading candidates
        
    Returns:
        List of candidates with assigned heading levels
    """
    classifier = HeadingLevelClassifier()
    classified = classifier.classify_heading_levels(candidates)
    return classifier.validate_level_assignments(classified)


def get_headings_by_level(candidates: List[HeadingCandidate], level: int) -> List[HeadingCandidate]:
    """
    Get all headings of a specific level.
    
    Args:
        candidates: List of classified candidates
        level: Heading level to filter (1, 2, or 3)
        
    Returns:
        List of candidates with the specified level
    """
    return [
        candidate for candidate in candidates
        if candidate.formatting_features.get('assigned_level') == level
    ]


def analyze_level_distribution(candidates: List[HeadingCandidate]) -> Dict[str, Any]:
    """
    Analyze the distribution of heading levels in a document.
    
    Args:
        candidates: List of classified candidates
        
    Returns:
        Dictionary with level distribution analysis
    """
    levels = [c.formatting_features.get('assigned_level', 0) for c in candidates]
    level_counts = Counter(levels)
    
    total_headings = len([l for l in levels if l in [1, 2, 3]])
    
    analysis = {
        'total_headings': total_headings,
        'level_counts': dict(level_counts),
        'level_percentages': {},
        'has_hierarchical_structure': False,
        'sequence_pattern': levels
    }
    
    # Calculate percentages
    if total_headings > 0:
        for level in [1, 2, 3]:
            count = level_counts.get(level, 0)
            analysis['level_percentages'][level] = (count / total_headings) * 100
    
    # Check for hierarchical structure
    has_h1 = level_counts.get(1, 0) > 0
    has_h2 = level_counts.get(2, 0) > 0
    has_h3 = level_counts.get(3, 0) > 0
    
    analysis['has_hierarchical_structure'] = has_h1 and (has_h2 or has_h3)
    
    return analysis