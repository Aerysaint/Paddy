"""
Unit tests for structure analyzer functionality.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from structure_analyzer import StructureAnalyzer, analyze_document_structure, get_high_confidence_headings
from data_models import TextBlock, HeadingCandidate


class TestStructureAnalyzer(unittest.TestCase):
    """Test cases for structure analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StructureAnalyzer()
        
        # Create sample text blocks for testing
        self.sample_blocks = [
            # Title-like text
            TextBlock("Document Title", 1, 16.0, "Arial-Bold", True, (100, 50, 300, 70), 20.0),
            
            # H1 heading with numbering
            TextBlock("1. Introduction", 1, 14.0, "Arial-Bold", True, (50, 100, 200, 120), 18.0),
            
            # Body text
            TextBlock("This is some body text that follows the introduction.", 1, 12.0, "Arial", False, (50, 130, 400, 145), 15.0),
            
            # H2 heading with numbering
            TextBlock("1.1 Background", 1, 13.0, "Arial-Bold", True, (50, 170, 180, 185), 16.0),
            
            # More body text
            TextBlock("More detailed body text explaining the background.", 1, 12.0, "Arial", False, (50, 195, 380, 210), 15.0),
            
            # H3 heading with numbering
            TextBlock("1.1.1 Historical Context", 1, 12.5, "Arial-Bold", True, (50, 240, 220, 255), 15.0),
            
            # H1 heading with keyword
            TextBlock("2. Methodology", 2, 14.0, "Arial-Bold", True, (50, 50, 200, 70), 18.0),
            
            # Non-numbered heading with keyword
            TextBlock("Results and Discussion", 2, 13.5, "Arial-Bold", True, (50, 100, 250, 115), 17.0),
            
            # All caps heading
            TextBlock("CONCLUSION", 3, 13.0, "Arial-Bold", True, (50, 50, 150, 65), 16.0),
            
            # Regular text that might look like heading but isn't
            TextBlock("3.5 million people were affected by this issue.", 3, 12.0, "Arial", False, (50, 100, 350, 115), 15.0),
        ]
    
    def test_analyzer_initialization(self):
        """Test StructureAnalyzer initialization."""
        self.assertIsInstance(self.analyzer, StructureAnalyzer)
        self.assertIn('primary', self.analyzer.heading_keywords)
        self.assertIn('secondary', self.analyzer.heading_keywords)
        self.assertIn('tertiary', self.analyzer.heading_keywords)
        self.assertIn('hierarchical_decimal', self.analyzer.numbering_patterns)
        self.assertEqual(len(self.analyzer.scoring_weights), 4)
    
    def test_analyze_document_structure(self):
        """Test document structure analysis."""
        candidates, context = self.analyzer.analyze_document_structure(self.sample_blocks)
        
        # Should return candidates and context
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
        # Check that high-confidence headings are identified
        high_conf_candidates = [c for c in candidates if c.confidence_score >= 0.6]
        self.assertGreater(len(high_conf_candidates), 0)
        
        # Verify candidate structure
        for candidate in candidates:
            self.assertIsInstance(candidate, HeadingCandidate)
            self.assertIsInstance(candidate.text, str)
            self.assertGreater(len(candidate.text), 0)
            self.assertIsInstance(candidate.confidence_score, float)
            self.assertGreaterEqual(candidate.confidence_score, 0.0)
            self.assertLessEqual(candidate.confidence_score, 1.0)
            self.assertIsInstance(candidate.formatting_features, dict)
            self.assertIsInstance(candidate.level_indicators, list)
    
    def test_numbering_pattern_detection(self):
        """Test numbering pattern detection."""
        test_cases = [
            ("1. Introduction", "simple_decimal"),
            ("1.1 Background", "hierarchical_decimal"),
            ("1.1.1 Details", "hierarchical_decimal"),
            ("2 Methods", "simple_number"),
            ("A. First Point", "letters"),
            ("I. Roman Numeral", "roman_numerals"),
            ("(1) Parenthetical", "parenthetical"),
            ("No numbering here", None),
        ]
        
        for text, expected_pattern in test_cases:
            result = self.analyzer._detect_numbering_pattern(text)
            self.assertEqual(result, expected_pattern, f"Failed for text: '{text}'")
    
    def test_numbering_level_extraction(self):
        """Test hierarchical level extraction from numbering."""
        test_cases = [
            ("1. Introduction", 1),
            ("1.1 Background", 2),
            ("1.1.1 Details", 3),
            ("1.2.3.4 Deep Level", 4),
            ("A. Letter", 1),
            ("No numbering", 0),
        ]
        
        for text, expected_level in test_cases:
            result = self.analyzer._extract_numbering_level(text)
            self.assertEqual(result, expected_level, f"Failed for text: '{text}'")
    
    def test_keyword_detection(self):
        """Test heading keyword detection."""
        test_cases = [
            ("Introduction to the Topic", ['introduction']),
            ("Chapter 1: Background", ['chapter']),
            ("Results and Discussion", ['results', 'discussion']),
            ("Methodology Overview", ['methodology']),
            ("Regular body text", []),
        ]
        
        for text, expected_keywords in test_cases:
            matches = self.analyzer._find_keyword_matches(text)
            found_keywords = []
            for category in matches.values():
                found_keywords.extend(category)
            
            for keyword in expected_keywords:
                self.assertIn(keyword, found_keywords, f"Keyword '{keyword}' not found in '{text}'")
    
    def test_confidence_scoring(self):
        """Test confidence scoring for different text types."""
        # Create test blocks with different characteristics
        test_blocks = [
            # High confidence: numbered, bold, isolated
            TextBlock("1. Introduction", 1, 14.0, "Arial-Bold", True, (50, 100, 200, 120), 18.0),
            
            # Medium confidence: keyword but no numbering
            TextBlock("Background Information", 1, 13.0, "Arial-Bold", True, (50, 150, 250, 170), 16.0),
            
            # Low confidence: regular body text
            TextBlock("This is regular body text with no special formatting.", 1, 12.0, "Arial", False, (50, 200, 400, 215), 15.0),
        ]
        
        context = self.analyzer._build_analysis_context(test_blocks)
        
        for i, block in enumerate(test_blocks):
            features = self.analyzer._extract_formatting_features(block, test_blocks, i, context)
            confidence = self.analyzer._calculate_confidence_score(block, features, context)
            
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
            # First block should have highest confidence
            if i == 0:
                self.assertGreater(confidence, 0.6, "Numbered heading should have high confidence")
    
    def test_visual_isolation_detection(self):
        """Test visual isolation detection."""
        # Create blocks with different spacing
        isolated_block = TextBlock("Isolated Heading", 1, 14.0, "Arial-Bold", True, (50, 100, 200, 120), 18.0)
        prev_block = TextBlock("Previous text", 1, 12.0, "Arial", False, (50, 50, 150, 65), 15.0)
        next_block = TextBlock("Following text", 1, 12.0, "Arial", False, (50, 150, 200, 165), 15.0)
        
        blocks = [prev_block, isolated_block, next_block]
        
        # Test isolation detection
        is_isolated = self.analyzer._is_visually_isolated(isolated_block, blocks, 1)
        self.assertIsInstance(is_isolated, bool)
        
        # Test whitespace calculation
        whitespace_before = self.analyzer._calculate_whitespace_before(isolated_block, blocks, 1)
        whitespace_after = self.analyzer._calculate_whitespace_after(isolated_block, blocks, 1)
        
        self.assertIsInstance(whitespace_before, float)
        self.assertIsInstance(whitespace_after, float)
        self.assertGreaterEqual(whitespace_before, 0.0)
        self.assertGreaterEqual(whitespace_after, 0.0)
    
    def test_text_structure_analysis(self):
        """Test text structure analysis features."""
        test_cases = [
            ("Title Case Heading", True, False),
            ("ALL CAPS HEADING", False, True),
            ("regular sentence case", False, False),
            ("Mixed Case With Some CAPS", True, False),  # This should be True since most words are capitalized
        ]
        
        for text, expected_title_case, expected_all_caps in test_cases:
            is_title_case = self.analyzer._is_title_case(text)
            is_all_caps = text.isupper() and len(text) > 1
            
            self.assertEqual(is_title_case, expected_title_case, f"Title case test failed for: '{text}'")
            self.assertEqual(is_all_caps, expected_all_caps, f"All caps test failed for: '{text}'")
    
    def test_confidence_thresholds(self):
        """Test confidence threshold classification."""
        candidates, _ = self.analyzer.analyze_document_structure(self.sample_blocks)
        
        # Test high confidence filtering
        high_conf = self.analyzer.get_high_confidence_candidates(candidates)
        for candidate in high_conf:
            self.assertGreaterEqual(candidate.confidence_score, self.analyzer.confidence_thresholds['high_confidence'])
        
        # Test ambiguous candidate filtering
        ambiguous = self.analyzer.get_ambiguous_candidates(candidates)
        min_conf, max_conf = self.analyzer.confidence_thresholds['ambiguous_range']
        for candidate in ambiguous:
            self.assertGreaterEqual(candidate.confidence_score, min_conf)
            self.assertLess(candidate.confidence_score, max_conf)
    
    def test_level_indicator_detection(self):
        """Test level indicator detection."""
        # Create a block with multiple indicators
        block = TextBlock("1.1 Introduction to Methods", 1, 14.0, "Arial-Bold", True, (50, 50, 250, 70), 18.0)
        context = self.analyzer._build_analysis_context([block])
        features = self.analyzer._extract_formatting_features(block, [block], 0, context)
        
        indicators = self.analyzer._detect_level_indicators(block, features)
        
        self.assertIsInstance(indicators, list)
        self.assertGreater(len(indicators), 0)
        
        # Should detect numbering level
        self.assertTrue(any('numbering_level' in indicator for indicator in indicators))
        
        # Should detect bold text
        self.assertIn('bold_text', indicators)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test analyze_document_structure function
        candidates, context = analyze_document_structure(self.sample_blocks)
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
        # Test get_high_confidence_headings function
        high_conf_headings = get_high_confidence_headings(self.sample_blocks)
        self.assertIsInstance(high_conf_headings, list)
        
        # All returned headings should be high confidence
        for heading in high_conf_headings:
            self.assertGreaterEqual(heading.confidence_score, 0.6)
    
    def test_empty_input_handling(self):
        """Test handling of empty or invalid input."""
        # Test with empty list
        candidates, context = self.analyzer.analyze_document_structure([])
        self.assertEqual(len(candidates), 0)
        
        # Test with blocks containing only empty text
        empty_blocks = [
            TextBlock("", 1, 12.0, "Arial", False, (0, 0, 0, 0), 15.0),
            TextBlock("   ", 1, 12.0, "Arial", False, (0, 0, 0, 0), 15.0),
        ]
        candidates, context = self.analyzer.analyze_document_structure(empty_blocks)
        self.assertEqual(len(candidates), 0)
    
    def test_multilingual_text_handling(self):
        """Test handling of multilingual text."""
        multilingual_blocks = [
            # Japanese text
            TextBlock("第1章 はじめに", 1, 14.0, "Arial", True, (50, 50, 200, 70), 18.0),
            
            # French text with accents
            TextBlock("1. Méthodologie", 1, 14.0, "Arial-Bold", True, (50, 100, 200, 120), 18.0),
            
            # German text with umlauts
            TextBlock("Übersicht", 1, 13.0, "Arial-Bold", True, (50, 150, 150, 170), 16.0),
        ]
        
        candidates, context = self.analyzer.analyze_document_structure(multilingual_blocks)
        
        # Should handle multilingual text without errors
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
        # Text should be preserved correctly
        for candidate in candidates:
            self.assertIsInstance(candidate.text, str)
            self.assertGreater(len(candidate.text), 0)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        edge_case_blocks = [
            # Very short text
            TextBlock("A", 1, 14.0, "Arial-Bold", True, (50, 50, 60, 70), 18.0),
            
            # Very long text
            TextBlock("This is a very long piece of text that might be mistaken for a heading but is actually body text because it contains too many words and characters to be a typical heading.", 1, 12.0, "Arial", False, (50, 100, 500, 130), 15.0),
            
            # Text with special characters
            TextBlock("§ 1.1 Legal Framework", 1, 13.0, "Arial-Bold", True, (50, 150, 200, 170), 16.0),
            
            # Text with numbers but not heading-like
            TextBlock("The study included 1,234 participants from 5 different countries.", 1, 12.0, "Arial", False, (50, 200, 400, 215), 15.0),
        ]
        
        candidates, context = self.analyzer.analyze_document_structure(edge_case_blocks)
        
        # Should handle edge cases without errors
        self.assertIsInstance(candidates, list)
        
        # Very short text might be detected as heading
        short_text_candidates = [c for c in candidates if c.text == "A"]
        if short_text_candidates:
            self.assertGreater(short_text_candidates[0].confidence_score, 0.0)
        
        # Very long text should have low confidence
        long_text_candidates = [c for c in candidates if "very long piece of text" in c.text]
        if long_text_candidates:
            self.assertLess(long_text_candidates[0].confidence_score, 0.5)


if __name__ == '__main__':
    unittest.main()