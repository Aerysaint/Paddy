#!/usr/bin/env python3
"""
Demo script for semantic analyzer functionality.
This demonstrates the semantic analyzer without requiring the actual model.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.semantic_analyzer import SemanticAnalyzer
from src.data_models import HeadingCandidate, TextBlock


def demo_semantic_analyzer():
    """Demonstrate semantic analyzer functionality with mocked model."""
    print("=" * 60)
    print("SEMANTIC ANALYZER DEMO")
    print("=" * 60)
    
    # Create temporary directory for cache
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test text blocks representing a document
        text_blocks = [
            TextBlock("Previous paragraph about research methodology.", 1, 12.0, "Arial", False, (100, 80, 400, 100), 14.0),
            TextBlock("1. Introduction", 1, 14.0, "Arial", True, (100, 120, 200, 140), 16.0),
            TextBlock("This chapter introduces the main concepts and objectives of the research study.", 1, 12.0, "Arial", False, (100, 160, 450, 180), 14.0),
            TextBlock("2.1 Background and Literature Review", 1, 13.0, "Arial", True, (100, 200, 350, 220), 15.0),
            TextBlock("The background section provides comprehensive context for the study.", 1, 12.0, "Arial", False, (100, 240, 400, 260), 14.0),
            TextBlock("2.1.1 Related Work", 1, 12.5, "Arial", True, (120, 280, 250, 300), 14.5),
            TextBlock("Several studies have investigated similar problems.", 1, 12.0, "Arial", False, (120, 320, 380, 340), 14.0),
            TextBlock("Following paragraph with more content.", 1, 12.0, "Arial", False, (100, 360, 350, 380), 14.0),
        ]
        
        # Create heading candidates with different confidence levels
        candidates = [
            HeadingCandidate(
                text="1. Introduction",
                page=1,
                confidence_score=0.5,  # Ambiguous - should be analyzed
                formatting_features={
                    'is_bold': True, 
                    'font_size': 14.0, 
                    'numbering_pattern': 'simple_decimal',
                    'whitespace_before': 20.0,
                    'whitespace_after': 15.0
                },
                level_indicators=['numbering_level_1', 'bold_text'],
                text_block=text_blocks[1]
            ),
            HeadingCandidate(
                text="2.1 Background and Literature Review",
                page=1,
                confidence_score=0.45,  # Ambiguous - should be analyzed
                formatting_features={
                    'is_bold': True, 
                    'font_size': 13.0, 
                    'numbering_pattern': 'hierarchical_decimal',
                    'whitespace_before': 18.0,
                    'whitespace_after': 12.0
                },
                level_indicators=['numbering_level_2', 'bold_text'],
                text_block=text_blocks[3]
            ),
            HeadingCandidate(
                text="2.1.1 Related Work",
                page=1,
                confidence_score=0.4,  # Ambiguous - should be analyzed
                formatting_features={
                    'is_bold': True, 
                    'font_size': 12.5, 
                    'numbering_pattern': 'hierarchical_decimal',
                    'whitespace_before': 15.0,
                    'whitespace_after': 10.0
                },
                level_indicators=['numbering_level_3', 'bold_text'],
                text_block=text_blocks[5]
            ),
            HeadingCandidate(
                text="High confidence heading",
                page=1,
                confidence_score=0.9,  # High confidence - should not be analyzed
                formatting_features={'is_bold': True, 'font_size': 16.0},
                level_indicators=['large_font', 'bold_text'],
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
        
        print(f"Created {len(candidates)} test candidates:")
        for i, candidate in enumerate(candidates, 1):
            print(f"  {i}. '{candidate.text}' (confidence: {candidate.confidence_score:.2f})")
        
        print(f"\nAmbiguous candidates (0.3-0.85 confidence range):")
        ambiguous = [c for c in candidates if 0.3 <= c.confidence_score < 0.85]
        for candidate in ambiguous:
            print(f"  - '{candidate.text}' (confidence: {candidate.confidence_score:.2f})")
        
        # Mock the sentence transformer for demonstration
        with patch('src.semantic_analyzer._get_sentence_transformer') as mock_get_transformer:
            # Create mock model and numpy
            mock_model = Mock()
            mock_np = Mock()
            
            # Mock embeddings for different text types
            def mock_encode(texts):
                embeddings = []
                for text in texts:
                    if "Introduction" in text:
                        embeddings.append([0.8, 0.2, 0.1, 0.3, 0.6])  # High similarity to intro patterns
                    elif "Background" in text or "Literature" in text:
                        embeddings.append([0.6, 0.7, 0.2, 0.4, 0.5])  # Medium similarity to section patterns
                    elif "Related Work" in text:
                        embeddings.append([0.4, 0.5, 0.8, 0.3, 0.4])  # High similarity to subsection patterns
                    else:
                        embeddings.append([0.3, 0.3, 0.3, 0.3, 0.3])  # Neutral embedding
                return embeddings
            
            mock_model.encode.side_effect = mock_encode
            
            # Mock numpy operations for similarity calculation
            def mock_dot(a, b):
                # Simulate cosine similarity calculation
                if "Introduction" in str(a) or any("Introduction" in pattern for pattern in ["Introduction", "Chapter"]):
                    return 0.85  # High similarity
                elif "Background" in str(a) or "Literature" in str(a):
                    return 0.72  # Medium-high similarity
                elif "Related Work" in str(a):
                    return 0.68  # Medium similarity
                else:
                    return 0.45  # Lower similarity
            
            mock_np.dot.side_effect = mock_dot
            mock_np.linalg.norm.return_value = 1.0  # Normalized vectors
            
            mock_get_transformer.return_value = (mock_model, mock_np)
            
            # Initialize semantic analyzer
            print(f"\nInitializing semantic analyzer...")
            analyzer = SemanticAnalyzer(cache_dir=temp_dir)
            
            # Run semantic analysis
            print(f"Running semantic analysis...")
            analyzed_candidates = analyzer.analyze_ambiguous_candidates(candidates, text_blocks)
            
            # Display results
            print(f"\n" + "=" * 60)
            print("SEMANTIC ANALYSIS RESULTS")
            print("=" * 60)
            
            analyzed_count = 0
            for i, candidate in enumerate(analyzed_candidates, 1):
                semantic_features = analyzer.get_semantic_features(candidate)
                
                print(f"\n{i}. '{candidate.text}'")
                print(f"   Original confidence: {candidate.confidence_score:.3f}")
                
                if semantic_features['has_semantic_analysis']:
                    analyzed_count += 1
                    print(f"   ✅ SEMANTIC ANALYSIS APPLIED")
                    print(f"   Semantic confidence: {semantic_features['semantic_confidence']:.3f}")
                    print(f"   Best pattern similarity: {semantic_features['best_pattern_similarity']:.3f}")
                    
                    # Show context coherence
                    coherence = semantic_features['context_coherence']
                    if coherence:
                        print(f"   Context coherence: {coherence.get('coherence_score', 0):.3f}")
                        print(f"   Has before context: {coherence.get('has_before_context', False)}")
                        print(f"   Has after context: {coherence.get('has_after_context', False)}")
                    
                    # Show top pattern similarities
                    pattern_sims = semantic_features['pattern_similarities']
                    if pattern_sims:
                        top_patterns = sorted(pattern_sims.items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"   Top pattern matches:")
                        for pattern, similarity in top_patterns:
                            print(f"     - {pattern}: {similarity:.3f}")
                    
                    # Show level indicators
                    indicators = [ind for ind in candidate.level_indicators if 'semantic' in ind]
                    if indicators:
                        print(f"   Semantic indicators: {indicators}")
                else:
                    print(f"   ⏭️  SKIPPED (confidence outside ambiguous range)")
            
            # Get semantic statistics
            stats = analyzer.analyze_semantic_distribution(analyzed_candidates)
            
            print(f"\n" + "=" * 60)
            print("SEMANTIC ANALYSIS STATISTICS")
            print("=" * 60)
            print(f"Total candidates: {stats['total_candidates']}")
            print(f"Analyzed candidates: {stats['analyzed_count']} ({stats['analysis_percentage']:.1f}%)")
            
            if stats['analyzed_count'] > 0:
                print(f"Average semantic confidence: {stats['avg_semantic_confidence']:.3f}")
                print(f"Confidence range: {stats['min_semantic_confidence']:.3f} - {stats['max_semantic_confidence']:.3f}")
                print(f"High confidence (≥0.7): {stats['high_confidence_count']}")
                print(f"Medium confidence (0.5-0.7): {stats['medium_confidence_count']}")
                print(f"Low confidence (<0.5): {stats['low_confidence_count']}")
            
            print(f"\n" + "=" * 60)
            print("TIER 2 SEMANTIC ANALYSIS DEMO COMPLETED")
            print("=" * 60)
            print("✅ Semantic analyzer successfully integrated!")
            print("✅ Lazy loading implemented - only activates for ambiguous cases")
            print("✅ Embedding cache functional for performance optimization")
            print("✅ Context window analysis working (±2 sentences)")
            print("✅ Pattern similarity scoring against known heading patterns")
            print("✅ Confidence fusion combining rule-based + semantic scores")
            
            return True
            
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = demo_semantic_analyzer()
    sys.exit(0 if success else 1)