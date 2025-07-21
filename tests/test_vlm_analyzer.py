"""
Tests for VLM analyzer module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_models import TextBlock, HeadingCandidate
from vlm_analyzer import VLMAnalyzer, VisualFeatures, VLMContext, analyze_highest_ambiguity_candidates


class TestVLMAnalyzer(unittest.TestCase):
    """Test cases for VLM analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = VLMAnalyzer()
        
        # Create sample text blocks
        self.text_blocks = [
            TextBlock(
                text="Chapter 1: Introduction",
                page=1,
                font_size=16.0,
                font_name="Arial-Bold",
                is_bold=True,
                bbox=(50.0, 100.0, 200.0, 120.0),
                line_height=20.0
            ),
            TextBlock(
                text="This is the introduction paragraph with some content.",
                page=1,
                font_size=12.0,
                font_name="Arial",
                is_bold=False,
                bbox=(50.0, 140.0, 400.0, 160.0),
                line_height=16.0
            ),
            TextBlock(
                text="1.1 Background",
                page=1,
                font_size=14.0,
                font_name="Arial-Bold",
                is_bold=True,
                bbox=(50.0, 180.0, 150.0, 200.0),
                line_height=18.0
            )
        ]
        
        # Create sample heading candidates
        self.candidates = [
            HeadingCandidate(
                text="Chapter 1: Introduction",
                page=1,
                confidence_score=0.4,  # Low confidence to trigger VLM
                formatting_features={
                    'is_large_font': True,
                    'is_bold': True,
                    'whitespace_before': 20.0,
                    'whitespace_after': 15.0,
                    'is_isolated': True,
                    'page_position': 0.2,
                    'semantic_confidence': 0.3
                },
                level_indicators=['large_font', 'bold_text'],
                text_block=self.text_blocks[0]
            ),
            HeadingCandidate(
                text="1.1 Background",
                page=1,
                confidence_score=0.3,  # Low confidence to trigger VLM
                formatting_features={
                    'numbering_pattern': 'hierarchical_decimal',
                    'is_bold': True,
                    'whitespace_before': 10.0,
                    'whitespace_after': 8.0,
                    'is_isolated': False,
                    'page_position': 0.4
                },
                level_indicators=['numbering_level_2'],
                text_block=self.text_blocks[2]
            )
        ]
    
    def test_filter_high_ambiguity_candidates(self):
        """Test filtering of high ambiguity candidates."""
        high_ambiguity = self.analyzer._filter_high_ambiguity_candidates(self.candidates)
        
        # Both candidates should be filtered as high ambiguity (confidence < 0.5)
        self.assertEqual(len(high_ambiguity), 2)
        self.assertIn(self.candidates[0], high_ambiguity)
        self.assertIn(self.candidates[1], high_ambiguity)
    
    def test_detect_text_visual_conflict(self):
        """Test detection of text-visual conflicts."""
        # Test large font with low semantic confidence (conflict)
        conflict_candidate = HeadingCandidate(
            text="Test Heading",
            page=1,
            confidence_score=0.5,
            formatting_features={
                'is_large_font': True,
                'semantic_confidence': 0.2  # Low semantic, high visual
            },
            level_indicators=[],
            text_block=self.text_blocks[0]
        )
        
        has_conflict = self.analyzer._detect_text_visual_conflict(conflict_candidate)
        self.assertTrue(has_conflict)
        
        # Test no conflict case
        no_conflict_candidate = HeadingCandidate(
            text="Test Heading",
            page=1,
            confidence_score=0.5,
            formatting_features={
                'is_large_font': True,
                'semantic_confidence': 0.8  # High semantic, high visual
            },
            level_indicators=[],
            text_block=self.text_blocks[0]
        )
        
        has_conflict = self.analyzer._detect_text_visual_conflict(no_conflict_candidate)
        self.assertFalse(has_conflict)
    
    def test_build_vlm_context(self):
        """Test building VLM context."""
        context = self.analyzer._build_vlm_context(self.candidates, self.text_blocks)
        
        self.assertIsInstance(context, VLMContext)
        self.assertEqual(len(context.document_text_blocks), 3)
        self.assertIn(1, context.page_dimensions)
        self.assertIn('avg_font_size', context.document_stats)
        self.assertIn('avg_spacing', context.document_stats)
    
    def test_extract_visual_features(self):
        """Test visual feature extraction."""
        context = self.analyzer._build_vlm_context(self.candidates, self.text_blocks)
        visual_features = self.analyzer._extract_visual_features(self.candidates[0], context)
        
        self.assertIsInstance(visual_features, VisualFeatures)
        self.assertEqual(len(visual_features.bbox_normalized), 4)
        self.assertTrue(visual_features.is_bold)
        self.assertTrue(visual_features.is_isolated)
        self.assertGreater(visual_features.font_size_ratio, 1.0)  # Larger than average
        
        # Test feature vector conversion
        feature_vector = visual_features.to_vector()
        self.assertEqual(len(feature_vector), 13)  # Expected number of features
        self.assertIsInstance(feature_vector, list)
        self.assertTrue(all(isinstance(x, (int, float)) for x in feature_vector))
    
    def test_extract_visual_features_no_text_block(self):
        """Test visual feature extraction with no text block."""
        candidate_no_block = HeadingCandidate(
            text="Test",
            page=1,
            confidence_score=0.5,
            formatting_features={},
            level_indicators=[],
            text_block=None
        )
        
        context = self.analyzer._build_vlm_context([], self.text_blocks)
        visual_features = self.analyzer._extract_visual_features(candidate_no_block, context)
        
        # Should return default features
        self.assertEqual(visual_features.bbox_normalized, (0.0, 0.0, 0.0, 0.0))
        self.assertFalse(visual_features.is_bold)
        self.assertFalse(visual_features.is_isolated)
        self.assertEqual(visual_features.font_size_ratio, 1.0)
    
    @patch('src.vlm_analyzer._get_torch_modules')
    def test_initialize_tokenizer(self, mock_torch):
        """Test tokenizer initialization."""
        # Mock torch modules
        mock_torch.return_value = (Mock(), Mock(), Mock(), Mock())
        
        self.analyzer._initialize_tokenizer()
        
        self.assertIsNotNone(self.analyzer.char_to_id)
        self.assertIsNotNone(self.analyzer.id_to_char)
        self.assertGreater(self.analyzer.vocab_size, 0)
        self.assertIn('<PAD>', self.analyzer.char_to_id)
        self.assertIn('<UNK>', self.analyzer.char_to_id)
    
    @patch('src.vlm_analyzer._get_torch_modules')
    def test_tokenize_text(self, mock_torch):
        """Test text tokenization."""
        # Mock torch modules
        mock_torch.return_value = (Mock(), Mock(), Mock(), Mock())
        
        self.analyzer._initialize_tokenizer()
        
        # Test normal text
        tokens = self.analyzer._tokenize_text("Hello", max_length=10)
        self.assertEqual(len(tokens), 10)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        
        # Test empty text
        tokens = self.analyzer._tokenize_text("", max_length=5)
        self.assertEqual(len(tokens), 5)
        self.assertTrue(all(t == self.analyzer.pad_token_id for t in tokens))
    
    @patch('src.vlm_analyzer._get_torch_modules')
    def test_model_size_estimate(self, mock_torch):
        """Test model size estimation."""
        # Mock torch modules and model
        mock_torch_module = Mock()
        mock_nn = Mock()
        mock_f = Mock()
        mock_np = Mock()
        mock_torch.return_value = (mock_torch_module, mock_nn, mock_f, mock_np)
        
        # Test without model loaded
        size_info = self.analyzer.get_model_size_estimate()
        self.assertFalse(size_info['model_loaded'])
        
        # Mock model with parameters
        mock_model = Mock()
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000
        mock_param1.requires_grad = True
        mock_param2 = Mock()
        mock_param2.numel.return_value = 500
        mock_param2.requires_grad = True
        
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        self.analyzer.model = mock_model
        
        size_info = self.analyzer.get_model_size_estimate()
        self.assertTrue(size_info['model_loaded'])
        self.assertEqual(size_info['total_parameters'], 1500)
        self.assertEqual(size_info['trainable_parameters'], 1500)
        self.assertAlmostEqual(size_info['estimated_size_mb'], 1500 * 4 / (1024 * 1024), places=6)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test that convenience functions exist and are callable
        self.assertTrue(callable(analyze_highest_ambiguity_candidates))
        
        # Test actual functionality (VLM will modify candidates)
        result = analyze_highest_ambiguity_candidates(self.candidates, self.text_blocks)
        
        # Should return same number of candidates
        self.assertEqual(len(result), len(self.candidates))
        
        # Check that VLM analysis was applied to high ambiguity candidates
        vlm_analyzed_count = sum(
            1 for c in result 
            if c.formatting_features.get('vlm_analysis_applied', False)
        )
        self.assertGreater(vlm_analyzed_count, 0)
        
        # Confidence scores should be updated for analyzed candidates
        for original, updated in zip(self.candidates, result):
            if updated.formatting_features.get('vlm_analysis_applied', False):
                # VLM analysis should have modified the confidence
                self.assertNotEqual(original.confidence_score, updated.confidence_score)


class TestVisualFeatures(unittest.TestCase):
    """Test cases for VisualFeatures dataclass."""
    
    def test_visual_features_creation(self):
        """Test VisualFeatures creation and methods."""
        features = VisualFeatures(
            bbox_normalized=(0.1, 0.2, 0.3, 0.4),
            spacing_before=0.5,
            spacing_after=0.6,
            font_size_ratio=1.2,
            is_bold=True,
            is_isolated=False,
            page_position=0.3,
            horizontal_alignment=0.1,
            text_density=0.8,
            aspect_ratio=2.5
        )
        
        # Test to_vector method
        vector = features.to_vector()
        expected_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 1.0, 0.0, 0.3, 0.1, 0.8, 2.5]
        self.assertEqual(vector, expected_vector)
        self.assertEqual(len(vector), 13)


class TestVLMContext(unittest.TestCase):
    """Test cases for VLMContext dataclass."""
    
    def test_vlm_context_creation(self):
        """Test VLMContext creation."""
        text_blocks = [
            TextBlock("Test", 1, 12.0, "Arial", False, (0, 0, 100, 20), 16.0)
        ]
        
        context = VLMContext(
            document_text_blocks=text_blocks,
            page_dimensions={1: (612.0, 792.0)},
            document_stats={'avg_font_size': 12.0},
            visual_features_cache={}
        )
        
        self.assertEqual(len(context.document_text_blocks), 1)
        self.assertEqual(context.page_dimensions[1], (612.0, 792.0))
        self.assertEqual(context.document_stats['avg_font_size'], 12.0)
        self.assertEqual(len(context.visual_features_cache), 0)


if __name__ == '__main__':
    unittest.main()