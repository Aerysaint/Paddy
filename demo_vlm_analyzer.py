#!/usr/bin/env python3
"""
Demo script for VLM analyzer functionality.

This script demonstrates the Vision-Language Model integration for PDF outline extraction,
showing how the VLM analyzer processes high ambiguity candidates and provides multimodal
analysis combining text content with visual layout features.
"""

import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_models import TextBlock, HeadingCandidate
from vlm_analyzer import VLMAnalyzer, analyze_highest_ambiguity_candidates, get_vlm_statistics, get_model_info
from logging_config import setup_logging

# Set up logging
logger = setup_logging()


def create_sample_data():
    """Create sample text blocks and heading candidates for testing."""
    
    # Create sample text blocks representing a document
    text_blocks = [
        TextBlock(
            text="Document Title: Advanced PDF Processing",
            page=1,
            font_size=18.0,
            font_name="Arial-Bold",
            is_bold=True,
            bbox=(50.0, 50.0, 400.0, 75.0),
            line_height=25.0
        ),
        TextBlock(
            text="This document describes advanced techniques for PDF processing and analysis.",
            page=1,
            font_size=12.0,
            font_name="Arial",
            is_bold=False,
            bbox=(50.0, 100.0, 500.0, 120.0),
            line_height=16.0
        ),
        TextBlock(
            text="1. Introduction",
            page=1,
            font_size=16.0,
            font_name="Arial-Bold",
            is_bold=True,
            bbox=(50.0, 150.0, 200.0, 170.0),
            line_height=20.0
        ),
        TextBlock(
            text="The introduction provides background information about PDF processing challenges.",
            page=1,
            font_size=12.0,
            font_name="Arial",
            is_bold=False,
            bbox=(50.0, 180.0, 480.0, 200.0),
            line_height=16.0
        ),
        TextBlock(
            text="1.1 Background",
            page=1,
            font_size=14.0,
            font_name="Arial-Bold",
            is_bold=True,
            bbox=(70.0, 220.0, 200.0, 240.0),
            line_height=18.0
        ),
        TextBlock(
            text="Background information about the field and current approaches.",
            page=1,
            font_size=12.0,
            font_name="Arial",
            is_bold=False,
            bbox=(70.0, 250.0, 450.0, 270.0),
            line_height=16.0
        ),
        TextBlock(
            text="2. Methodology",
            page=2,
            font_size=16.0,
            font_name="Arial-Bold",
            is_bold=True,
            bbox=(50.0, 50.0, 200.0, 70.0),
            line_height=20.0
        ),
        TextBlock(
            text="This section describes the methodology used in our approach.",
            page=2,
            font_size=12.0,
            font_name="Arial",
            is_bold=False,
            bbox=(50.0, 80.0, 460.0, 100.0),
            line_height=16.0
        ),
        TextBlock(
            text="Conclusion",
            page=3,
            font_size=15.0,
            font_name="Arial-Bold",
            is_bold=True,
            bbox=(50.0, 50.0, 150.0, 70.0),
            line_height=19.0
        )
    ]
    
    # Create heading candidates with varying confidence levels
    candidates = [
        HeadingCandidate(
            text="Document Title: Advanced PDF Processing",
            page=1,
            confidence_score=0.9,  # High confidence - won't trigger VLM
            formatting_features={
                'is_large_font': True,
                'is_bold': True,
                'whitespace_before': 0.0,
                'whitespace_after': 25.0,
                'is_isolated': True,
                'page_position': 0.1,
                'semantic_confidence': 0.85,
                'semantic_analysis_applied': True
            },
            level_indicators=['large_font', 'bold_text', 'high_semantic_confidence'],
            text_block=text_blocks[0]
        ),
        HeadingCandidate(
            text="1. Introduction",
            page=1,
            confidence_score=0.4,  # Low confidence - will trigger VLM
            formatting_features={
                'numbering_pattern': 'simple_decimal',
                'is_bold': True,
                'whitespace_before': 30.0,
                'whitespace_after': 10.0,
                'is_isolated': True,
                'page_position': 0.3,
                'semantic_confidence': 0.6,
                'semantic_analysis_applied': True
            },
            level_indicators=['numbering_level_1', 'bold_text'],
            text_block=text_blocks[2]
        ),
        HeadingCandidate(
            text="1.1 Background",
            page=1,
            confidence_score=0.3,  # Low confidence - will trigger VLM
            formatting_features={
                'numbering_pattern': 'hierarchical_decimal',
                'is_bold': True,
                'whitespace_before': 20.0,
                'whitespace_after': 10.0,
                'is_isolated': False,
                'page_position': 0.5,
                'semantic_confidence': 0.4,
                'semantic_analysis_applied': True
            },
            level_indicators=['numbering_level_2', 'bold_text'],
            text_block=text_blocks[4]
        ),
        HeadingCandidate(
            text="2. Methodology",
            page=2,
            confidence_score=0.45,  # Borderline confidence - will trigger VLM
            formatting_features={
                'numbering_pattern': 'simple_decimal',
                'is_bold': True,
                'whitespace_before': 0.0,  # Top of page
                'whitespace_after': 10.0,
                'is_isolated': True,
                'page_position': 0.1,
                'semantic_confidence': 0.7,
                'semantic_analysis_applied': True,
                'is_large_font': True  # Conflict: large font but low overall confidence
            },
            level_indicators=['numbering_level_1', 'bold_text', 'large_font'],
            text_block=text_blocks[6]
        ),
        HeadingCandidate(
            text="Conclusion",
            page=3,
            confidence_score=0.35,  # Low confidence - will trigger VLM
            formatting_features={
                'is_bold': True,
                'whitespace_before': 0.0,
                'whitespace_after': 20.0,
                'is_isolated': True,
                'page_position': 0.1,
                'semantic_confidence': 0.8,  # Conflict: high semantic but low overall
                'semantic_analysis_applied': True,
                'keyword_matches': {'primary': ['conclusion']}
            },
            level_indicators=['bold_text', 'primary_keywords'],
            text_block=text_blocks[8]
        )
    ]
    
    return text_blocks, candidates


def demonstrate_vlm_analysis():
    """Demonstrate VLM analysis functionality."""
    
    print("=== VLM Analyzer Demo ===\n")
    
    # Create sample data
    text_blocks, candidates = create_sample_data()
    
    print(f"Created {len(text_blocks)} text blocks and {len(candidates)} heading candidates")
    print("\nOriginal candidates:")
    for i, candidate in enumerate(candidates):
        print(f"  {i+1}. '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
    
    # Initialize VLM analyzer
    print("\n=== Initializing VLM Analyzer ===")
    analyzer = VLMAnalyzer()
    
    # Check model info before loading
    print("\nModel info before initialization:")
    model_info = analyzer.get_model_size_estimate()
    print(f"  Model loaded: {model_info['model_loaded']}")
    
    # Filter high ambiguity candidates
    print("\n=== Filtering High Ambiguity Candidates ===")
    high_ambiguity = analyzer._filter_high_ambiguity_candidates(candidates)
    print(f"Found {len(high_ambiguity)} high ambiguity candidates:")
    for candidate in high_ambiguity:
        print(f"  - '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
    
    # Check for text-visual conflicts
    print("\n=== Checking for Text-Visual Conflicts ===")
    for candidate in candidates:
        has_conflict = analyzer._detect_text_visual_conflict(candidate)
        if has_conflict:
            print(f"  Conflict detected in: '{candidate.text}'")
    
    # Build VLM context
    print("\n=== Building VLM Context ===")
    context = analyzer._build_vlm_context(candidates, text_blocks)
    print(f"  Document stats: {context.document_stats}")
    print(f"  Page dimensions: {context.page_dimensions}")
    
    # Extract visual features for high ambiguity candidates
    print("\n=== Extracting Visual Features ===")
    for candidate in high_ambiguity[:2]:  # Show first 2 for brevity
        visual_features = analyzer._extract_visual_features(candidate, context)
        print(f"  '{candidate.text}':")
        print(f"    Font size ratio: {visual_features.font_size_ratio:.2f}")
        print(f"    Is bold: {visual_features.is_bold}")
        print(f"    Is isolated: {visual_features.is_isolated}")
        print(f"    Page position: {visual_features.page_position:.2f}")
        print(f"    Spacing before: {visual_features.spacing_before:.3f}")
        print(f"    Feature vector length: {len(visual_features.to_vector())}")
    
    # Perform VLM analysis
    print("\n=== Performing VLM Analysis ===")
    try:
        analyzed_candidates = analyzer.analyze_highest_ambiguity_candidates(candidates, text_blocks)
        
        print("Analysis completed successfully!")
        print("\nUpdated candidates:")
        for i, candidate in enumerate(analyzed_candidates):
            vlm_applied = candidate.formatting_features.get('vlm_analysis_applied', False)
            vlm_conf = candidate.formatting_features.get('vlm_confidence', 0.0)
            print(f"  {i+1}. '{candidate.text}' (confidence: {candidate.confidence_score:.3f})")
            if vlm_applied:
                print(f"      VLM confidence: {vlm_conf:.3f}")
        
        # Get VLM statistics
        print("\n=== VLM Analysis Statistics ===")
        stats = analyzer.analyze_vlm_distribution(analyzed_candidates)
        if stats['analyzed_count'] > 0:
            print(f"  Candidates analyzed: {stats['analyzed_count']}/{stats['total_candidates']}")
            print(f"  Analysis percentage: {stats['analysis_percentage']:.1f}%")
            print(f"  Average VLM confidence: {stats['avg_vlm_confidence']:.3f}")
            print(f"  High confidence count: {stats['high_confidence_count']}")
            print(f"  Medium confidence count: {stats['medium_confidence_count']}")
            print(f"  Low confidence count: {stats['low_confidence_count']}")
        else:
            print("  No candidates were analyzed with VLM")
        
        # Get model size information
        print("\n=== Model Size Information ===")
        model_info = analyzer.get_model_size_estimate()
        if model_info['model_loaded']:
            print(f"  Total parameters: {model_info['total_parameters']:,}")
            print(f"  Estimated size: {model_info['estimated_size_mb']:.2f} MB")
            print(f"  Within size limit (35MB): {model_info['within_size_limit']}")
            print(f"  Vocabulary size: {model_info['vocab_size']:,}")
        
    except Exception as e:
        print(f"VLM analysis failed: {e}")
        logger.error(f"VLM analysis error: {e}", exc_info=True)
        
        # Show what would happen without actual model loading
        print("\nSimulating VLM analysis results...")
        for candidate in high_ambiguity:
            # Simulate VLM confidence boost
            simulated_vlm_conf = min(0.8, candidate.confidence_score + 0.2)
            simulated_final_conf = candidate.confidence_score * 0.7 + simulated_vlm_conf * 0.3
            print(f"  '{candidate.text}': {candidate.confidence_score:.3f} -> {simulated_final_conf:.3f}")


def test_convenience_functions():
    """Test convenience functions."""
    
    print("\n=== Testing Convenience Functions ===")
    
    text_blocks, candidates = create_sample_data()
    
    try:
        # Test convenience function
        print("Testing analyze_highest_ambiguity_candidates()...")
        result = analyze_highest_ambiguity_candidates(candidates, text_blocks)
        print(f"  Returned {len(result)} candidates")
        
        # Test statistics function
        print("Testing get_vlm_statistics()...")
        stats = get_vlm_statistics(result)
        print(f"  Statistics: {stats}")
        
        # Test model info function
        print("Testing get_model_info()...")
        info = get_model_info()
        print(f"  Model info: {info}")
        
    except Exception as e:
        print(f"Convenience function test failed: {e}")
        logger.error(f"Convenience function error: {e}", exc_info=True)


def main():
    """Main demo function."""
    
    print("VLM Analyzer Demo - Vision-Language Model Integration")
    print("=" * 60)
    
    try:
        # Run main demonstration
        demonstrate_vlm_analysis()
        
        # Test convenience functions
        test_convenience_functions()
        
        print("\n=== Demo Completed Successfully ===")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)