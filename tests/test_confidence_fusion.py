"""
Tests for the hybrid confidence fusion system.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.confidence_fusion import (
    ConfidenceFusionSystem, DocumentTypeDetector, DocumentType,
    ConfidenceWeights, analyze_document_with_fusion, get_high_confidence_headings
)
from src.data_models import TextBlock, HeadingCandidate


class TestDocumentTypeDetector:
    """Test document type detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DocumentTypeDetector()
    
    def test_detect_academic_document(self):
        """Test detection of academic document type."""
        text_blocks = [
            TextBlock("Abstract", 1, 14.0, "Arial", True, (100, 100, 200, 120), 16.0),
            TextBlock("This paper presents a methodology for analyzing...", 1, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
            TextBlock("1. Introduction", 1, 13.0, "Arial", True, (100, 160, 250, 180), 15.0),
            TextBlock("The research hypothesis is that...", 1, 12.0, "Arial", False, (100, 190, 500, 210), 14.0),
            TextBlock("References", 5, 13.0, "Arial", True, (100, 100, 200, 120), 15.0),
        ]
        
        doc_type = self.detector.detect_document_type(text_blocks)
        assert doc_type == DocumentType.ACADEMIC
    
    def test_detect_technical_document(self):
        """Test detection of technical document type."""
        text_blocks = [
            TextBlock("System Architecture", 1, 14.0, "Arial", True, (100, 100, 300, 120), 16.0),
            TextBlock("This specification defines the API interface...", 1, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
            TextBlock("1.1 Implementation Details", 1, 13.0, "Arial", True, (100, 160, 350, 180), 15.0),
            TextBlock("The function parameters are...", 1, 12.0, "Arial", False, (100, 190, 500, 210), 14.0),
            TextBlock("Configuration", 2, 13.0, "Arial", True, (100, 100, 250, 120), 15.0),
        ]
        
        doc_type = self.detector.detect_document_type(text_blocks)
        assert doc_type == DocumentType.TECHNICAL
    
    def test_detect_business_document(self):
        """Test detection of business document type."""
        text_blocks = [
            TextBlock("Executive Summary", 1, 14.0, "Arial", True, (100, 100, 300, 120), 16.0),
            TextBlock("The quarterly revenue increased by 15%...", 1, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
            TextBlock("Financial Analysis", 2, 13.0, "Arial", True, (100, 160, 300, 180), 15.0),
            TextBlock("Budget allocation for Q4 shows $2M investment...", 2, 12.0, "Arial", False, (100, 190, 500, 210), 14.0),
            TextBlock("Market Strategy", 3, 13.0, "Arial", True, (100, 100, 250, 120), 15.0),
        ]
        
        doc_type = self.detector.detect_document_type(text_blocks)
        assert doc_type == DocumentType.BUSINESS
    
    def test_detect_legal_document(self):
        """Test detection of legal document type."""
        text_blocks = [
            TextBlock("Article 1", 1, 14.0, "Arial", True, (100, 100, 200, 120), 16.0),
            TextBlock("Whereas the parties agree to the following terms...", 1, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
            TextBlock("Section 2.1", 1, 13.0, "Arial", True, (100, 160, 200, 180), 15.0),
            TextBlock("Therefore, pursuant to this agreement...", 1, 12.0, "Arial", False, (100, 190, 500, 210), 14.0),
            TextBlock("Clause 3", 2, 13.0, "Arial", True, (100, 100, 150, 120), 15.0),
        ]
        
        doc_type = self.detector.detect_document_type(text_blocks)
        assert doc_type == DocumentType.LEGAL
    
    def test_detect_unknown_document(self):
        """Test detection when document type is unclear."""
        text_blocks = [
            TextBlock("Some random text", 1, 12.0, "Arial", False, (100, 100, 300, 120), 14.0),
            TextBlock("More random content without clear indicators", 1, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
        ]
        
        doc_type = self.detector.detect_document_type(text_blocks)
        assert doc_type == DocumentType.UNKNOWN
    
    def test_empty_document(self):
        """Test handling of empty document."""
        doc_type = self.detector.detect_document_type([])
        assert doc_type == DocumentType.UNKNOWN


class TestConfidenceWeights:
    """Test confidence weights functionality."""
    
    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = ConfidenceWeights(0.6, 0.3, 0.3)
        normalized = weights.normalize()
        
        assert abs(normalized.rule_based + normalized.semantic + normalized.visual - 1.0) < 1e-6
        assert normalized.rule_based == 0.5  # 0.6 / 1.2
        assert normalized.semantic == 0.25   # 0.3 / 1.2
        assert normalized.visual == 0.25     # 0.3 / 1.2
    
    def test_normalize_zero_weights(self):
        """Test normalization with zero weights."""
        weights = ConfidenceWeights(0.0, 0.0, 0.0)
        normalized = weights.normalize()
        
        assert normalized.rule_based == 1.0
        assert normalized.semantic == 0.0
        assert normalized.visual == 0.0


class TestConfidenceFusionSystem:
    """Test the main confidence fusion system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fusion_system = ConfidenceFusionSystem()
        
        # Create sample text blocks
        self.text_blocks = [
            TextBlock("1. Introduction", 1, 14.0, "Arial", True, (100, 100, 250, 120), 16.0),
            TextBlock("This document presents...", 1, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
            TextBlock("1.1 Background", 1, 13.0, "Arial", True, (100, 160, 280, 180), 15.0),
            TextBlock("The background information...", 1, 12.0, "Arial", False, (100, 190, 500, 210), 14.0),
            TextBlock("2. Methodology", 2, 14.0, "Arial", True, (100, 100, 280, 120), 16.0),
            TextBlock("Our approach involves...", 2, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
        ]
        
        # Create sample heading candidates
        self.candidates = [
            HeadingCandidate(
                text="1. Introduction",
                page=1,
                confidence_score=0.9,
                formatting_features={'numbering_pattern': 'simple_decimal', 'is_bold': True},
                level_indicators=['numbering_level_1'],
                text_block=self.text_blocks[0]
            ),
            HeadingCandidate(
                text="1.1 Background",
                page=1,
                confidence_score=0.6,
                formatting_features={'numbering_pattern': 'hierarchical_decimal', 'is_bold': True},
                level_indicators=['numbering_level_2'],
                text_block=self.text_blocks[2]
            ),
            HeadingCandidate(
                text="2. Methodology",
                page=2,
                confidence_score=0.4,
                formatting_features={'numbering_pattern': 'simple_decimal', 'is_bold': True},
                level_indicators=['numbering_level_1'],
                text_block=self.text_blocks[4]
            ),
        ]
    
    @patch('src.confidence_fusion.StructureAnalyzer')
    def test_tier1_rule_based_analysis(self, mock_analyzer_class):
        """Test Tier 1 rule-based analysis."""
        # Mock the structure analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze_document_structure.return_value = (self.candidates, Mock())
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create processing stats
        from src.confidence_fusion import ProcessingStats
        stats = ProcessingStats(0, 0, 0, 0, 0, 0.0, {'tier1': 0, 'tier2': 0, 'tier3': 0})
        
        # Test Tier 1 analysis
        result_candidates = self.fusion_system._tier1_rule_based_analysis(self.text_blocks, stats)
        
        assert len(result_candidates) == 3
        assert stats.tier_distribution['tier1'] == 3
        assert stats.high_confidence_count == 1  # Only first candidate has >0.85 confidence
        mock_analyzer.analyze_document_structure.assert_called_once_with(self.text_blocks)
    
    @patch('src.confidence_fusion.SemanticAnalyzer')
    def test_tier2_semantic_analysis(self, mock_analyzer_class):
        """Test Tier 2 semantic analysis."""
        # Mock the semantic analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze_ambiguous_candidates.return_value = self.candidates
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create processing stats
        from src.confidence_fusion import ProcessingStats
        stats = ProcessingStats(0, 0, 0, 0, 0, 0.0, {'tier1': 0, 'tier2': 0, 'tier3': 0})
        
        # Create weights with semantic component
        weights = ConfidenceWeights(0.5, 0.3, 0.2)
        
        # Test Tier 2 analysis
        result_candidates = self.fusion_system._tier2_semantic_analysis(
            self.candidates, self.text_blocks, weights, stats
        )
        
        assert len(result_candidates) == 3
        assert stats.semantic_analysis_count == 2  # Two candidates in ambiguous range (0.3-0.85)
        mock_analyzer.analyze_ambiguous_candidates.assert_called_once()
    
    @patch('src.confidence_fusion.VLMAnalyzer')
    def test_tier3_vlm_analysis(self, mock_analyzer_class):
        """Test Tier 3 VLM analysis."""
        # Mock the VLM analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze_highest_ambiguity_candidates.return_value = self.candidates
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create processing stats
        from src.confidence_fusion import ProcessingStats
        stats = ProcessingStats(0, 0, 0, 0, 0, 0.0, {'tier1': 0, 'tier2': 0, 'tier3': 0})
        
        # Create weights with visual component
        weights = ConfidenceWeights(0.5, 0.2, 0.3)
        
        # Test Tier 3 analysis
        result_candidates = self.fusion_system._tier3_vlm_analysis(
            self.candidates, self.text_blocks, weights, stats
        )
        
        assert len(result_candidates) == 3
        # VLM analysis is limited to max 10 candidates and sorted by confidence
        # Only the lowest confidence candidate (0.4) should be analyzed
        assert stats.vlm_analysis_count == 1  # One candidate below VLM threshold (0.5)
        mock_analyzer.analyze_highest_ambiguity_candidates.assert_called_once()
    
    def test_apply_final_fusion(self):
        """Test final confidence fusion."""
        # Add semantic and visual scores to candidates
        self.candidates[0].formatting_features.update({
            'semantic_confidence': 0.8,
            'vlm_confidence': 0.7
        })
        self.candidates[1].formatting_features.update({
            'semantic_confidence': 0.6,
            'vlm_confidence': 0.5
        })
        self.candidates[2].formatting_features.update({
            'semantic_confidence': 0.4,
            'vlm_confidence': 0.3
        })
        
        weights = ConfidenceWeights(0.5, 0.3, 0.2)
        
        # Test final fusion
        fused_candidates = self.fusion_system._apply_final_fusion(
            self.candidates, weights, DocumentType.ACADEMIC
        )
        
        assert len(fused_candidates) == 3
        
        # Check that fusion was applied
        for candidate in fused_candidates:
            assert candidate.formatting_features.get('fusion_applied') is True
            assert 'rule_based_score' in candidate.formatting_features
            assert 'semantic_score' in candidate.formatting_features
            assert 'visual_score' in candidate.formatting_features
            assert 'final_fused_score' in candidate.formatting_features
    
    def test_document_type_adjustments(self):
        """Test document type specific adjustments."""
        base_score = 0.6
        candidate = self.candidates[0]  # Has numbering pattern
        
        # Test academic adjustment - should boost for numbered sections
        adjusted_score = self.fusion_system._apply_document_type_adjustments(
            base_score, candidate, DocumentType.ACADEMIC
        )
        assert adjusted_score >= base_score  # Should be boosted or same for numbered sections
        
        # Test technical adjustment - should boost for hierarchical numbering
        candidate_hierarchical = self.candidates[1]  # Has hierarchical numbering
        adjusted_score = self.fusion_system._apply_document_type_adjustments(
            base_score, candidate_hierarchical, DocumentType.TECHNICAL
        )
        assert adjusted_score > base_score  # Should be boosted for hierarchical numbering
    
    def test_performance_constraint_checking(self):
        """Test performance constraint monitoring."""
        # Set start time
        self.fusion_system.start_time = time.time()
        
        # Should not exceed constraint immediately
        assert not self.fusion_system._check_performance_constraint()
        
        # Mock time passage
        self.fusion_system.start_time = time.time() - 15.0  # 15 seconds ago
        assert self.fusion_system._check_performance_constraint()
    
    @patch('src.confidence_fusion.StructureAnalyzer')
    def test_full_analysis_workflow(self, mock_analyzer_class):
        """Test the complete analysis workflow."""
        # Mock structure analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze_document_structure.return_value = (self.candidates, Mock())
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock the document type detector on the fusion system instance
        mock_detector = Mock()
        mock_detector.detect_document_type.return_value = DocumentType.ACADEMIC
        self.fusion_system.document_type_detector = mock_detector
        
        # Run full analysis
        result = self.fusion_system.analyze_document_with_fusion(self.text_blocks)
        
        # Verify result structure
        assert result.document_type == DocumentType.ACADEMIC
        assert len(result.candidates) == 3
        assert 'total_processing_time' in result.performance_metrics
        assert 'total_candidates' in result.processing_stats
        assert result.performance_metrics['within_target'] is not None
    
    def test_confidence_distribution_calculation(self):
        """Test confidence distribution calculation."""
        # Create candidates with different confidence levels
        candidates = [
            HeadingCandidate("High", 1, 0.9, {}, []),
            HeadingCandidate("Medium", 1, 0.7, {}, []),
            HeadingCandidate("Low", 1, 0.4, {}, []),
        ]
        
        distribution = self.fusion_system._calculate_confidence_distribution(candidates)
        
        assert distribution['high'] == 1
        assert distribution['medium'] == 1
        assert distribution['low'] == 1


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.text_blocks = [
            TextBlock("1. Introduction", 1, 14.0, "Arial", True, (100, 100, 250, 120), 16.0),
            TextBlock("Some content...", 1, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
        ]
    
    @patch('src.confidence_fusion.ConfidenceFusionSystem')
    def test_analyze_document_with_fusion(self, mock_fusion_class):
        """Test the convenience function for document analysis."""
        # Mock fusion system
        mock_fusion = Mock()
        mock_result = Mock()
        mock_result.candidates = []
        mock_result.document_type = DocumentType.UNKNOWN
        mock_result.processing_stats = {}
        mock_result.performance_metrics = {}
        
        mock_fusion.analyze_document_with_fusion.return_value = mock_result
        mock_fusion_class.return_value = mock_fusion
        
        # Test function
        result = analyze_document_with_fusion(self.text_blocks)
        
        assert result == mock_result
        mock_fusion.analyze_document_with_fusion.assert_called_once_with(self.text_blocks)
    
    @patch('src.confidence_fusion.analyze_document_with_fusion')
    def test_get_high_confidence_headings(self, mock_analyze):
        """Test the convenience function for getting high-confidence headings."""
        # Mock analysis result
        mock_result = Mock()
        mock_result.candidates = [
            HeadingCandidate("High", 1, 0.9, {}, []),
            HeadingCandidate("Low", 1, 0.4, {}, []),
        ]
        mock_analyze.return_value = mock_result
        
        # Test function
        high_conf_headings = get_high_confidence_headings(self.text_blocks, 0.8)
        
        assert len(high_conf_headings) == 1
        assert high_conf_headings[0].text == "High"
        assert high_conf_headings[0].confidence_score == 0.9


class TestIntegration:
    """Integration tests for the confidence fusion system."""
    
    def test_realistic_document_processing(self):
        """Test processing a realistic document structure."""
        # Create a realistic document with various heading types
        text_blocks = [
            # Title
            TextBlock("Document Analysis Report", 1, 16.0, "Arial", True, (100, 50, 400, 80), 18.0),
            
            # H1 headings
            TextBlock("1. Executive Summary", 1, 14.0, "Arial", True, (100, 120, 350, 140), 16.0),
            TextBlock("The analysis shows significant improvements...", 1, 12.0, "Arial", False, (100, 150, 500, 170), 14.0),
            
            TextBlock("2. Introduction", 1, 14.0, "Arial", True, (100, 200, 280, 220), 16.0),
            TextBlock("This report presents findings from...", 1, 12.0, "Arial", False, (100, 230, 500, 250), 14.0),
            
            # H2 headings
            TextBlock("2.1 Background", 2, 13.0, "Arial", True, (120, 100, 280, 120), 15.0),
            TextBlock("Previous studies have shown...", 2, 12.0, "Arial", False, (120, 130, 500, 150), 14.0),
            
            TextBlock("2.2 Objectives", 2, 13.0, "Arial", True, (120, 180, 280, 200), 15.0),
            TextBlock("The primary objectives are...", 2, 12.0, "Arial", False, (120, 210, 500, 230), 14.0),
            
            # H3 heading
            TextBlock("2.2.1 Specific Goals", 2, 12.5, "Arial", True, (140, 260, 320, 280), 14.5),
            TextBlock("The specific goals include...", 2, 12.0, "Arial", False, (140, 290, 500, 310), 14.0),
            
            # H1 heading
            TextBlock("3. Methodology", 3, 14.0, "Arial", True, (100, 100, 300, 120), 16.0),
            TextBlock("Our methodology consists of...", 3, 12.0, "Arial", False, (100, 130, 500, 150), 14.0),
        ]
        
        # This test would require mocking the analyzers to avoid dependencies
        # For now, we'll test that the system can handle the structure
        fusion_system = ConfidenceFusionSystem()
        
        # Test document type detection
        doc_type = fusion_system.document_type_detector.detect_document_type(text_blocks)
        assert doc_type in [DocumentType.ACADEMIC, DocumentType.BUSINESS, DocumentType.TECHNICAL, DocumentType.UNKNOWN]
        
        # Test that weights are properly configured
        weights = fusion_system.type_weights[doc_type]
        assert isinstance(weights, ConfidenceWeights)
        assert weights.rule_based > 0
        assert weights.semantic >= 0
        assert weights.visual >= 0


if __name__ == "__main__":
    pytest.main([__file__])