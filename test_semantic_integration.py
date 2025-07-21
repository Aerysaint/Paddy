#!/usr/bin/env python3
"""
Integration test for semantic analyzer with actual sentence-transformers model.
This test verifies that the semantic analyzer can load and use the MiniLM-L6-v2 model.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.semantic_analyzer import SemanticAnalyzer, analyze_ambiguous_candidates
from src.data_models import HeadingCandidate, TextBlock


def test_semantic_analyzer_with_real_model():
    """Test semantic analyzer with actual sentence-transformers model."""
    print("Testing semantic analyzer with real sentence-transformers model...")
    
    # Create temporary directory for cache
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test text blocks
        text_blocks = [
            TextBlock("Previous paragraph about methodology.", 1, 12.0, "Arial", False, (100, 80, 300, 100), 14.0),
            TextBlock("1. Introduction", 1, 14.0, "Arial", True, (100, 120, 200, 140), 16.0),
            TextBlock("This chapter introduces the main concepts and objectives of the research.", 1, 12.0, "Arial", False, (100, 160, 400, 180), 14.0),
            TextBlock("2.1 Background", 1, 13.0, "Arial", True, (100, 200, 220, 220), 15.0),
            TextBlock("The background section provides context for the study.", 1, 12.0, "Arial", False, (100, 240, 350, 260), 14.0),
            TextBlock("Following paragraph text.", 1, 12.0, "Arial", False, (100, 280, 250, 300), 14.0),
        ]
        
        # Create heading candidates with different confidence levels
        candidates = [
            HeadingCandidate(
                text="1. Introduction",
                page=1,
                confidence_score=0.5,  # Ambiguous - should be analyzed
                formatting_features={'is_bold': True, 'font_size': 14.0, 'numbering_pattern': 'simple_decimal'},
                level_indicators=['numbering_level_1', 'bold_text'],
                text_block=text_blocks[1]
            ),
            HeadingCandidate(
                text="2.1 Background",
                page=1,
                confidence_score=0.45,  # Ambiguous - should be analyzed
                formatting_features={'is_bold': True, 'font_size': 13.0, 'numbering_pattern': 'hierarchical_decimal'},
                level_indicators=['numbering_level_2', 'bold_text'],
                text_block=text_blocks[3]
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
        
        print(f"Created {len(candidates)} test candidates")
        print(f"Ambiguous candidates (0.3-0.85): {[c.text for c in candidates if 0.3 <= c.confidence_score < 0.85]}")
        
        # Initialize semantic analyzer
        analyzer = SemanticAnalyzer(cache_dir=temp_dir)
        
        # Run semantic analysis
        print("\nRunning semantic analysis...")
        analyzed_candidates = analyzer.analyze_ambiguous_candidates(candidates, text_blocks)
        
        # Analyze results
        print(f"\nAnalysis complete. Results:")
        print(f"Total candidates: {len(analyzed_candidates)}")
        
        # Check which candidates were analyzed
        analyzed_count = 0
        for candidate in analyzed_candidates:
            semantic_features = analyzer.get_semantic_features(candidate)
            
            if semantic_features['has_semantic_analysis']:
                analyzed_count += 1
                print(f"\nAnalyzed candidate: '{candidate.text}'")
                print(f"  Original confidence: {candidate.confidence_score:.3f}")
                print(f"  Semantic confidence: {semantic_features['semantic_confidence']:.3f}")
                print(f"  Best pattern similarity: {semantic_features['best_pattern_similarity']:.3f}")
                print(f"  Context coherence: {semantic_features['context_coherence'].get('coherence_score', 0):.3f}")
                
                # Show top pattern similarities
                pattern_sims = semantic_features['pattern_similarities']
                if pattern_sims:
                    top_patterns = sorted(pattern_sims.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"  Top pattern matches: {top_patterns}")
        
        print(f"\nSemantic analysis applied to {analyzed_count} candidates")
        
        # Get semantic statistics
        stats = analyzer.analyze_semantic_distribution(analyzed_candidates)
        print(f"\nSemantic Analysis Statistics:")
        print(f"  Analyzed: {stats['analyzed_count']}/{stats['total_candidates']} ({stats['analysis_percentage']:.1f}%)")
        if stats['analyzed_count'] > 0:
            print(f"  Avg semantic confidence: {stats['avg_semantic_confidence']:.3f}")
            print(f"  Range: {stats['min_semantic_confidence']:.3f} - {stats['max_semantic_confidence']:.3f}")
            print(f"  High confidence: {stats['high_confidence_count']}")
            print(f"  Medium confidence: {stats['medium_confidence_count']}")
            print(f"  Low confidence: {stats['low_confidence_count']}")
        
        # Test convenience function
        print("\nTesting convenience function...")
        convenience_result = analyze_ambiguous_candidates(candidates, text_blocks, temp_dir)
        print(f"Convenience function returned {len(convenience_result)} candidates")
        
        print("\n✅ Semantic analyzer integration test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        print("sentence-transformers is required for semantic analysis")
        return False
        
    except Exception as e:
        print(f"❌ Error during semantic analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_embedding_cache_functionality():
    """Test embedding cache functionality with real model."""
    print("\nTesting embedding cache functionality...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from src.semantic_analyzer import EmbeddingCache
        
        # Create cache
        cache = EmbeddingCache(cache_dir=temp_dir)
        
        # Test texts
        test_texts = [
            "Introduction",
            "Chapter 1",
            "Background",
            "Methodology",
            "Results and Discussion"
        ]
        
        print(f"Testing cache with {len(test_texts)} texts...")
        
        # First pass - should generate embeddings
        print("First pass (generating embeddings)...")
        for text in test_texts:
            embedding = cache.get_embedding(text)
            print(f"  '{text}': {'cached' if embedding is not None else 'not cached'}")
        
        # Save cache
        cache.save()
        
        # Create new cache instance
        new_cache = EmbeddingCache(cache_dir=temp_dir)
        
        # Second pass - should load from cache
        print("Second pass (loading from cache)...")
        cached_count = 0
        for text in test_texts:
            embedding = new_cache.get_embedding(text)
            if embedding is not None:
                cached_count += 1
            print(f"  '{text}': {'cached' if embedding is not None else 'not cached'}")
        
        print(f"Cache functionality test: {cached_count}/{len(test_texts)} texts cached")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing cache: {e}")
        return False
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC ANALYZER INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Main semantic analysis functionality
    success1 = test_semantic_analyzer_with_real_model()
    
    # Test 2: Embedding cache functionality
    success2 = test_embedding_cache_functionality()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)