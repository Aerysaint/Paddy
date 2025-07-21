"""
Tests for semantic analyzer module.
"""

import unittest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.semantic_analyzer import SemanticAnalyzer, EmbeddingCache, analyze_ambiguous_candidates
from src.data_models import HeadingCandidate, TextBlock


class TestEmbeddingCache(unittest.TestCase):
    """Test cases for EmbeddingCache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = EmbeddingCache(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_text_hash_generation(self):
        """Test text hash generation."""
        text1 = "Introduction"
        text2 = "Introduction"
        text3 = "Chapter 1"
        
        hash1 = self.cache.get_text_hash(text1)
        hash2 = self.cache.get_text_hash(text2)
        hash3 = self.cache.get_text_hash(text3)
        
        self.assertEqual(hash1, hash2)  # Same text should have same hash
        self.assertNotEqual(hash1, hash3)  # Different text should have different hash
        self.assertEqual(len(hash1), 32)  # MD5 hash length
    
    def test_cache_operations(self):
        """Test cache store and retrieve operations."""
        text = "Introduction"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Initially should return None
        self.assertIsNone(self.cache.get_embedding(text))
        
        # Store embedding
        self.cache.store_embedding(text, embedding)
        
        # Should now return the stored embedding
        retrieved = self.cache.get_embedding(text)
        self.assertEqual(retrieved, embedding)
    
    def test_cache_persistence(self):
        """Test cache persistence to disk."""
        text = "Chapter 1"
        embedding = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        # Store in first cache instance
        self.cache.store_embedding(text, embedding)
        self.cache.save()
        
        # Create new cache instance with same directory
        new_cache = EmbeddingCache(cache_dir=self.temp_dir)
        
        # Should load the previously stored embedding
        retrieved = new_cache.get_embedding(text)
        self.assertEqual(retrieved, embedding)


class TestSemanticAnalyzer(unittest.TestCase):
    """Test cases for SemanticAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = SemanticAnalyzer(cache_dir=self.temp_dir)
        
        # Create mock text blocks
        self.text_blocks = [
            TextBlock("Previous paragraph text.", 1, 12.0, "Arial", False, (100, 100, 200, 120), 14.0),
            TextBlock("1. Introduction", 1, 14.0, "Arial", True, (100, 140, 200, 160), 16.0),
            TextBlock("This is the introduction content.", 1, 12.0, "Arial", False, (100, 180, 300, 200), 14.0),
            TextBlock("Following paragraph text.", 1, 12.0, "Arial", False, (100, 220, 250, 240), 14.0),
        ]
        
        # Create mock heading candidates
        self.candidates = [
            HeadingCandidate(
                text="1. Introduction",
                page=1,
                confidence_score=0.5,  # Ambiguous confidence
                formatting_features={'is_bold': True, 'font_size': 14.0},
                level_indicators=['numbering_level_1'],
                text_block=self.text_blocks[1]
            ),
            HeadingCandidate(
                text="High confidence heading",
                page=1,
                confidence_score=0.9,  # High confidence - should not be analyzed
                formatting_features={'is_bold': True, 'font_size': 16.0},
                level_indicators=['large_font'],
                text_block=None
            ),
            HeadingCandidate(
                text="Low confidence text",
                page=1,
                confidence_score=0.2,  # Low confidence - should not be analyzed
                formatting_features={'is_bold': False, 'font_size': 12.0},
                level_indicators=[],
                text_block=None
            )
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_filter_ambiguous_candidates(self):
        """Test filtering of ambiguous candidates."""
        ambiguous = self.analyzer._filter_ambiguous_candidates(self.candidates)
        
        # Only the candidate with confidence 0.5 should be ambiguous
        self.assertEqual(len(ambiguous), 1)
        self.assertEqual(ambiguous[0].text, "1. Introduction")
        self.assertEqual(ambiguous[0].confidence_score, 0.5)
    
    def test_extract_context_window(self):
        """Test context window extraction."""
        candidate = self.candidates[0]  # "1. Introduction"
        
        before_context, after_context = self.analyzer._extract_context_window(
            candidate, self.text_blocks
        )
        
        # Should extract context from surrounding blocks
        self.assertIn("Previous paragraph", before_context)
        self.assertIn("introduction content", after_context)
        self.assertIn("Following paragraph", after_context)
    
    def test_get_candidate_id(self):
        """Test candidate ID generation."""
        candidate = self.candidates[0]
        candidate_id = self.analyzer._get_candidate_id(candidate)
        
        # Should generate a unique ID
        self.assertIsInstance(candidate_id, str)
        self.assertIn("1_1. Introduction", candidate_id)
    
    @patch('src.semantic_analyzer._get_sentence_transformer')
    def test_semantic_analysis_without_model(self, mock_get_transformer):
        """Test semantic analysis behavior when model is not available."""
        # Mock the transformer to raise ImportError
        mock_get_transformer.side_effect = ImportError("sentence-transformers not available")
        
        # Should handle the error gracefully
        with self.assertRaises(ImportError):
            self.analyzer.analyze_ambiguous_candidates(self.candidates, self.text_blocks)
    
    @patch('src.semantic_analyzer._get_sentence_transformer')
    def test_semantic_analysis_with_mock_model(self, mock_get_transformer):
        """Test semantic analysis with mocked model."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_np = Mock()
        
        # Mock embeddings
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_model.encode.return_value = [mock_embedding]
        
        # Mock numpy operations
        mock_np.dot.return_value = 0.8
        mock_np.linalg.norm.return_value = 1.0
        
        mock_get_transformer.return_value = (mock_model, mock_np)
        
        # Run semantic analysis
        result = self.analyzer.analyze_ambiguous_candidates(self.candidates, self.text_blocks)
        
        # Should return updated candidates
        self.assertEqual(len(result), len(self.candidates))
        
        # Find the analyzed candidate
        analyzed_candidate = None
        for candidate in result:
            if candidate.formatting_features.get('semantic_analysis_applied'):
                analyzed_candidate = candidate
                break
        
        self.assertIsNotNone(analyzed_candidate)
        self.assertTrue(analyzed_candidate.formatting_features.get('semantic_analysis_applied'))
        self.assertIn('semantic_confidence', analyzed_candidate.formatting_features)
    
    def test_semantic_features_extraction(self):
        """Test semantic features extraction from analyzed candidate."""
        # Create a candidate with semantic features
        candidate = HeadingCandidate(
            text="Test heading",
            page=1,
            confidence_score=0.6,
            formatting_features={
                'semantic_confidence': 0.75,
                'pattern_similarities': {'Introduction': 0.8, 'Chapter': 0.6},
                'context_coherence': {'coherence_score': 0.7},
                'semantic_analysis_applied': True
            },
            level_indicators=['medium_semantic_confidence'],
            text_block=None
        )
        
        features = self.analyzer.get_semantic_features(candidate)
        
        self.assertEqual(features['semantic_confidence'], 0.75)
        self.assertEqual(features['pattern_similarities'], {'Introduction': 0.8, 'Chapter': 0.6})
        self.assertEqual(features['context_coherence'], {'coherence_score': 0.7})
        self.assertTrue(features['has_semantic_analysis'])
        self.assertEqual(features['best_pattern_similarity'], 0.8)
    
    def test_semantic_distribution_analysis(self):
        """Test semantic distribution analysis."""
        # Create candidates with and without semantic analysis
        candidates = [
            HeadingCandidate(
                text="Analyzed 1", page=1, confidence_score=0.6,
                formatting_features={'semantic_analysis_applied': True, 'semantic_confidence': 0.8},
                level_indicators=[], text_block=None
            ),
            HeadingCandidate(
                text="Analyzed 2", page=1, confidence_score=0.7,
                formatting_features={'semantic_analysis_applied': True, 'semantic_confidence': 0.6},
                level_indicators=[], text_block=None
            ),
            HeadingCandidate(
                text="Not analyzed", page=1, confidence_score=0.9,
                formatting_features={}, level_indicators=[], text_block=None
            )
        ]
        
        stats = self.analyzer.analyze_semantic_distribution(candidates)
        
        self.assertEqual(stats['analyzed_count'], 2)
        self.assertEqual(stats['total_candidates'], 3)
        self.assertAlmostEqual(stats['analysis_percentage'], 66.67, places=1)
        self.assertEqual(stats['avg_semantic_confidence'], 0.7)
        self.assertEqual(stats['min_semantic_confidence'], 0.6)
        self.assertEqual(stats['max_semantic_confidence'], 0.8)


class TestSemanticAnalyzerIntegration(unittest.TestCase):
    """Integration tests for semantic analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_convenience_function(self):
        """Test convenience function for semantic analysis."""
        # Create test data
        text_blocks = [
            TextBlock("1. Introduction", 1, 14.0, "Arial", True, (100, 100, 200, 120), 16.0),
            TextBlock("Content here", 1, 12.0, "Arial", False, (100, 140, 300, 160), 14.0),
        ]
        
        candidates = [
            HeadingCandidate(
                text="1. Introduction",
                page=1,
                confidence_score=0.5,  # Ambiguous
                formatting_features={'is_bold': True},
                level_indicators=['numbering_level_1'],
                text_block=text_blocks[0]
            )
        ]
        
        # Test with mocked transformer
        with patch('src.semantic_analyzer._get_sentence_transformer') as mock_get_transformer:
            mock_model = Mock()
            mock_np = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_np.dot.return_value = 0.7
            mock_np.linalg.norm.return_value = 1.0
            mock_get_transformer.return_value = (mock_model, mock_np)
            
            result = analyze_ambiguous_candidates(candidates, text_blocks, self.temp_dir)
            
            self.assertEqual(len(result), 1)
            # Should have semantic analysis applied
            analyzed = result[0]
            self.assertTrue(analyzed.formatting_features.get('semantic_analysis_applied', False))


if __name__ == '__main__':
    unittest.main()