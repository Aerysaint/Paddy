"""
Tests for the heading level classifier module.

Tests cover level assignment based on numbering hierarchy, font size analysis,
edge cases handling, and non-hierarchical sequence support.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.heading_level_classifier import (
    HeadingLevelClassifier,
    LevelAssignmentContext,
    classify_heading_levels,
    get_headings_by_level,
    analyze_level_distribution
)
from src.data_models import HeadingCandidate, TextBlock


class TestHeadingLevelClassifier:
    """Test cases for HeadingLevelClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = HeadingLevelClassifier()
    
    def create_text_block(self, text: str, page: int = 1, font_size: float = 12.0, 
                         is_bold: bool = False, bbox: tuple = (0, 0, 100, 20)) -> TextBlock:
        """Helper to create TextBlock instances."""
        return TextBlock(
            text=text,
            page=page,
            font_size=font_size,
            font_name="Arial",
            is_bold=is_bold,
            bbox=bbox,
            line_height=font_size * 1.2
        )
    
    def create_heading_candidate(self, text: str, page: int = 1, confidence: float = 0.8,
                               font_size: float = 12.0, is_bold: bool = False,
                               features: Dict[str, Any] = None) -> HeadingCandidate:
        """Helper to create HeadingCandidate instances."""
        text_block = self.create_text_block(text, page, font_size, is_bold)
        
        default_features = {
            'text_length': len(text),
            'font_size_ratio': font_size / 12.0,
            'is_bold': is_bold,
            'whitespace_before': 10.0,
            'whitespace_after': 10.0,
            'page_position': 0.3
        }
        
        if features:
            default_features.update(features)
        
        return HeadingCandidate(
            text=text,
            page=page,
            confidence_score=confidence,
            formatting_features=default_features,
            level_indicators=[],
            text_block=text_block
        )
    
    def test_init(self):
        """Test classifier initialization."""
        assert self.classifier is not None
        pattern_names = [name for name, _ in self.classifier.numbering_patterns]
        assert 'hierarchical_decimal' in pattern_names
        assert 'numbering_hierarchy' in self.classifier.level_weights
        assert self.classifier.font_size_tolerance == 0.5
    
    def test_classify_empty_candidates(self):
        """Test classification with empty candidate list."""
        result = self.classifier.classify_heading_levels([])
        assert result == []
    
    def test_classify_single_candidate(self):
        """Test classification with single candidate."""
        candidate = self.create_heading_candidate("1. Introduction", font_size=14.0, is_bold=True)
        
        result = self.classifier.classify_heading_levels([candidate])
        
        assert len(result) == 1
        assert result[0].formatting_features['assigned_level'] == 1
        assert 'heading_level_1' in result[0].level_indicators
    
    def test_numbering_hierarchy_analysis(self):
        """Test numbering hierarchy analysis for level determination."""
        candidates = [
            self.create_heading_candidate("1. Chapter One"),
            self.create_heading_candidate("1.1 Section One"),
            self.create_heading_candidate("1.1.1 Subsection One"),
            self.create_heading_candidate("2. Chapter Two"),
            self.create_heading_candidate("2.1 Section Two")
        ]
        
        result = self.classifier.classify_heading_levels(candidates)
        
        # Check level assignments based on numbering
        levels = [c.formatting_features['assigned_level'] for c in result]
        assert levels == [1, 2, 3, 1, 2]
    
    def test_font_size_analysis(self):
        """Test font size relationship analysis."""
        candidates = [
            self.create_heading_candidate("Large Heading", font_size=18.0, is_bold=True),
            self.create_heading_candidate("Medium Heading", font_size=14.0, is_bold=True),
            self.create_heading_candidate("Small Heading", font_size=12.0, is_bold=False)
        ]
        
        result = self.classifier.classify_heading_levels(candidates)
        
        # Larger fonts should get higher priority for H1
        levels = [c.formatting_features['assigned_level'] for c in result]
        assert levels[0] == 1  # Largest font should be H1
        assert levels[1] in [1, 2]  # Medium font could be H1 or H2
        assert levels[2] in [2, 3]  # Smallest font should be H2 or H3
    
    def test_non_hierarchical_sequences(self):
        """Test support for non-hierarchical sequences (H1→H3→H2)."""
        candidates = [
            self.create_heading_candidate("1. Main Topic", font_size=16.0),
            self.create_heading_candidate("1.1.1 Specific Detail", font_size=10.0),  # Should be H3
            self.create_heading_candidate("1.2 Subtopic", font_size=12.0)  # Should be H2
        ]
        
        result = self.classifier.classify_heading_levels(candidates)
        
        # Should preserve non-hierarchical sequence
        levels = [c.formatting_features['assigned_level'] for c in result]
        assert levels == [1, 3, 2]  # H1 → H3 → H2 sequence preserved
    
    def test_orphaned_headings(self):
        """Test handling of orphaned headings (H3 without H2 parent)."""
        candidates = [
            self.create_heading_candidate("1. Main Topic"),
            self.create_heading_candidate("1.1.1 Orphaned Detail")  # H3 without H2
        ]
        
        result = self.classifier.classify_heading_levels(candidates)
        
        # Should still assign H3 level based on numbering
        levels = [c.formatting_features['assigned_level'] for c in result]
        assert levels == [1, 3]
    
    def test_repeated_levels(self):
        """Test handling of repeated heading levels."""
        candidates = [
            self.create_heading_candidate("1. First Chapter"),
            self.create_heading_candidate("2. Second Chapter"),
            self.create_heading_candidate("3. Third Chapter")
        ]
        
        result = self.classifier.classify_heading_levels(candidates)
        
        # All should be classified as H1
        levels = [c.formatting_features['assigned_level'] for c in result]
        assert all(level == 1 for level in levels)
    
    def test_missing_intermediate_levels(self):
        """Test handling of missing intermediate levels."""
        candidates = [
            self.create_heading_candidate("1. Chapter"),
            self.create_heading_candidate("1.1.1 Deep Subsection")  # Missing 1.1 level
        ]
        
        result = self.classifier.classify_heading_levels(candidates)
        
        # Should assign levels based on numbering depth
        levels = [c.formatting_features['assigned_level'] for c in result]
        assert levels == [1, 3]  # H1, H3 (no H2)
    
    def test_extract_numbering_info(self):
        """Test numbering information extraction."""
        test_cases = [
            ("1. Introduction", ('simple_decimal', ['1'], 'Introduction')),
            ("1.1 Section", ('hierarchical_decimal', ['1', '1'], 'Section')),
            ("1.1.1 Subsection", ('hierarchical_decimal', ['1', '1', '1'], 'Subsection')),
            ("I. Roman Numeral", ('roman_numerals', ['I'], 'Roman Numeral')),
            ("A. Letter", ('letters', ['A'], 'Letter')),
            ("(1) Parenthetical", ('parenthetical', ['1'], 'Parenthetical')),
            ("No numbering", None)
        ]
        
        for text, expected in test_cases:
            result = self.classifier._extract_numbering_info(text)
            assert result == expected, f"Failed for text: {text}"
    
    def test_cluster_font_sizes(self):
        """Test font size clustering."""
        font_sizes = [12.0, 12.2, 14.0, 14.1, 16.0, 16.5]
        
        clusters = self.classifier._cluster_font_sizes(font_sizes)
        
        # Should group similar sizes together
        assert len(set(clusters.values())) <= 3  # At most 3 clusters
        assert clusters[12.0] == clusters[12.2]  # Similar sizes in same cluster
        assert clusters[14.0] == clusters[14.1]  # Similar sizes in same cluster
    
    def test_semantic_importance_analysis(self):
        """Test semantic importance analysis."""
        context = LevelAssignmentContext(
            document_font_sizes=[12.0],
            average_font_size=12.0,
            font_size_clusters={12.0: 0},
            numbering_sequences={},
            page_order=[]
        )
        
        # Test H1 keywords
        h1_candidate = self.create_heading_candidate("Introduction")
        h1_scores = self.classifier._analyze_semantic_importance(h1_candidate, context)
        assert h1_scores[1] > h1_scores[2] and h1_scores[1] > h1_scores[3]
        
        # Test H2 keywords
        h2_candidate = self.create_heading_candidate("Analysis Section")
        h2_scores = self.classifier._analyze_semantic_importance(h2_candidate, context)
        assert h2_scores[2] >= h2_scores[1]
        
        # Test H3 keywords
        h3_candidate = self.create_heading_candidate("Example Case")
        h3_scores = self.classifier._analyze_semantic_importance(h3_candidate, context)
        assert h3_scores[3] >= h3_scores[2]
    
    def test_visual_prominence_analysis(self):
        """Test visual prominence analysis."""
        context = LevelAssignmentContext(
            document_font_sizes=[12.0],
            average_font_size=12.0,
            font_size_clusters={12.0: 0},
            numbering_sequences={},
            page_order=[]
        )
        
        # Test top-of-page positioning
        top_candidate = self.create_heading_candidate(
            "Top Heading",
            features={'page_position': 0.1, 'is_isolated': True, 'whitespace_before': 25}
        )
        
        scores = self.classifier._analyze_visual_prominence(top_candidate, context)
        assert scores[1] > 0.5  # Should favor H1 for prominent positioning
    
    def test_validate_level_assignments(self):
        """Test level assignment validation."""
        candidates = [
            self.create_heading_candidate("1. Chapter"),
            self.create_heading_candidate("1.1.1 Deep Section"),  # Skipped level
            self.create_heading_candidate("2. Another Chapter")
        ]
        
        # Assign levels manually for testing
        candidates[0].formatting_features['assigned_level'] = 1
        candidates[1].formatting_features['assigned_level'] = 3
        candidates[2].formatting_features['assigned_level'] = 1
        
        result = self.classifier.validate_level_assignments(candidates)
        
        # Should preserve assignments (no correction)
        levels = [c.formatting_features['assigned_level'] for c in result]
        assert levels == [1, 3, 1]
    
    def test_complex_document_structure(self):
        """Test classification with complex document structure."""
        candidates = [
            self.create_heading_candidate("Abstract", font_size=14.0, is_bold=True),
            self.create_heading_candidate("1. Introduction", font_size=16.0, is_bold=True),
            self.create_heading_candidate("1.1 Background", font_size=14.0, is_bold=True),
            self.create_heading_candidate("1.1.1 Historical Context", font_size=12.0),
            self.create_heading_candidate("1.2 Methodology", font_size=14.0, is_bold=True),
            self.create_heading_candidate("2. Results", font_size=16.0, is_bold=True),
            self.create_heading_candidate("2.1 Analysis", font_size=14.0, is_bold=True),
            self.create_heading_candidate("3. Conclusion", font_size=16.0, is_bold=True)
        ]
        
        result = self.classifier.classify_heading_levels(candidates)
        
        # Check that we get reasonable level assignments
        levels = [c.formatting_features['assigned_level'] for c in result]
        
        # Abstract should be H1 (semantic importance)
        assert levels[0] == 1
        
        # Numbered chapters should be H1
        assert levels[1] == 1  # 1. Introduction
        assert levels[5] == 1  # 2. Results
        assert levels[7] == 1  # 3. Conclusion
        
        # Numbered sections should be H2
        assert levels[2] == 2  # 1.1 Background
        assert levels[4] == 2  # 1.2 Methodology
        assert levels[6] == 2  # 2.1 Analysis
        
        # Numbered subsections should be H3
        assert levels[3] == 3  # 1.1.1 Historical Context


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def create_classified_candidate(self, text: str, level: int, page: int = 1) -> HeadingCandidate:
        """Helper to create classified candidate."""
        candidate = HeadingCandidate(
            text=text,
            page=page,
            confidence_score=0.8,
            formatting_features={'assigned_level': level},
            level_indicators=[f'heading_level_{level}'],
            text_block=TextBlock(text, page, 12.0, "Arial", False, (0, 0, 100, 20), 14.4)
        )
        return candidate
    
    def test_classify_heading_levels_function(self):
        """Test the convenience classify_heading_levels function."""
        candidates = [
            HeadingCandidate(
                text="1. Test Heading",
                page=1,
                confidence_score=0.8,
                formatting_features={},
                level_indicators=[],
                text_block=TextBlock("1. Test Heading", 1, 14.0, "Arial", True, (0, 0, 100, 20), 16.8)
            )
        ]
        
        result = classify_heading_levels(candidates)
        
        assert len(result) == 1
        assert 'assigned_level' in result[0].formatting_features
        assert result[0].formatting_features['assigned_level'] in [1, 2, 3]
    
    def test_get_headings_by_level(self):
        """Test filtering headings by level."""
        candidates = [
            self.create_classified_candidate("Heading 1", 1),
            self.create_classified_candidate("Heading 2", 2),
            self.create_classified_candidate("Another H1", 1),
            self.create_classified_candidate("Heading 3", 3)
        ]
        
        h1_headings = get_headings_by_level(candidates, 1)
        h2_headings = get_headings_by_level(candidates, 2)
        h3_headings = get_headings_by_level(candidates, 3)
        
        assert len(h1_headings) == 2
        assert len(h2_headings) == 1
        assert len(h3_headings) == 1
        
        assert all(c.formatting_features['assigned_level'] == 1 for c in h1_headings)
        assert all(c.formatting_features['assigned_level'] == 2 for c in h2_headings)
        assert all(c.formatting_features['assigned_level'] == 3 for c in h3_headings)
    
    def test_analyze_level_distribution(self):
        """Test level distribution analysis."""
        candidates = [
            self.create_classified_candidate("H1 One", 1),
            self.create_classified_candidate("H1 Two", 1),
            self.create_classified_candidate("H2 One", 2),
            self.create_classified_candidate("H3 One", 3)
        ]
        
        analysis = analyze_level_distribution(candidates)
        
        assert analysis['total_headings'] == 4
        assert analysis['level_counts'][1] == 2
        assert analysis['level_counts'][2] == 1
        assert analysis['level_counts'][3] == 1
        assert analysis['level_percentages'][1] == 50.0
        assert analysis['level_percentages'][2] == 25.0
        assert analysis['level_percentages'][3] == 25.0
        assert analysis['has_hierarchical_structure'] is True
        assert analysis['sequence_pattern'] == [1, 1, 2, 3]


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = HeadingLevelClassifier()
    
    def test_candidates_without_text_blocks(self):
        """Test classification of candidates without text blocks."""
        candidate = HeadingCandidate(
            text="Test Heading",
            page=1,
            confidence_score=0.8,
            formatting_features={},
            level_indicators=[],
            text_block=None  # No text block
        )
        
        result = self.classifier.classify_heading_levels([candidate])
        
        assert len(result) == 1
        assert 'assigned_level' in result[0].formatting_features
    
    def test_empty_text_candidates(self):
        """Test classification of candidates with empty text."""
        candidate = HeadingCandidate(
            text="",
            page=1,
            confidence_score=0.8,
            formatting_features={},
            level_indicators=[],
            text_block=TextBlock("", 1, 12.0, "Arial", False, (0, 0, 0, 0), 14.4)
        )
        
        result = self.classifier.classify_heading_levels([candidate])
        
        assert len(result) == 1
        assert 'assigned_level' in result[0].formatting_features
    
    def test_very_low_confidence_scores(self):
        """Test handling of very low confidence scores."""
        candidate = HeadingCandidate(
            text="1. Low Confidence Heading",
            page=1,
            confidence_score=0.1,  # Very low confidence
            formatting_features={},
            level_indicators=[],
            text_block=TextBlock("1. Low Confidence Heading", 1, 12.0, "Arial", False, (0, 0, 100, 20), 14.4)
        )
        
        result = self.classifier.classify_heading_levels([candidate])
        
        # Should still assign a level
        assert len(result) == 1
        assert 'assigned_level' in result[0].formatting_features
    
    def test_extreme_font_sizes(self):
        """Test handling of extreme font sizes."""
        candidates = [
            HeadingCandidate(
                text="Tiny Text",
                page=1,
                confidence_score=0.8,
                formatting_features={},
                level_indicators=[],
                text_block=TextBlock("Tiny Text", 1, 6.0, "Arial", False, (0, 0, 50, 10), 7.2)
            ),
            HeadingCandidate(
                text="Huge Text",
                page=1,
                confidence_score=0.8,
                formatting_features={},
                level_indicators=[],
                text_block=TextBlock("Huge Text", 1, 48.0, "Arial", True, (0, 0, 200, 60), 57.6)
            )
        ]
        
        result = self.classifier.classify_heading_levels(candidates)
        
        assert len(result) == 2
        # Huge text should likely be H1
        huge_text_level = next(c.formatting_features['assigned_level'] for c in result if c.text == "Huge Text")
        assert huge_text_level == 1


if __name__ == '__main__':
    pytest.main([__file__])