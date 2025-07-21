"""
Semantic analyzer module for PDF outline extraction.

This module provides Tier 2 semantic analysis using sentence-transformers MiniLM-L6-v2
for ambiguous heading classification cases. Implements embedding cache, context window
analysis, and semantic similarity scoring against known heading patterns.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import hashlib

from .data_models import HeadingCandidate, TextBlock
from .logging_config import setup_logging

logger = setup_logging()

# Lazy import for sentence-transformers to avoid loading unless needed
_sentence_transformer_model = None
_numpy = None


def _get_sentence_transformer():
    """Lazy load sentence transformer model."""
    global _sentence_transformer_model, _numpy
    
    if _sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            _numpy = np
            
            logger.info("Loading sentence-transformers MiniLM-L6-v2 model...")
            _sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import sentence-transformers: {e}")
            raise ImportError("sentence-transformers is required for semantic analysis")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise RuntimeError(f"Could not initialize semantic model: {e}")
    
    return _sentence_transformer_model, _numpy


@dataclass
class SemanticContext:
    """Context information for semantic analysis."""
    document_text_blocks: List[TextBlock]
    candidate_contexts: Dict[str, Tuple[str, str]]  # candidate_id -> (before_context, after_context)
    document_embeddings: Dict[str, Any]  # text_hash -> embedding
    heading_pattern_embeddings: Dict[str, Any]  # pattern_name -> embedding


class EmbeddingCache:
    """Cache for storing and retrieving text embeddings."""
    
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "semantic_cache.pkl")
        self.cache = self._load_cache()
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.debug(f"Loaded embedding cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
        
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved embedding cache with {len(self.cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[Any]:
        """Get embedding from cache."""
        text_hash = self.get_text_hash(text)
        return self.cache.get(text_hash)
    
    def store_embedding(self, text: str, embedding: Any) -> None:
        """Store embedding in cache."""
        text_hash = self.get_text_hash(text)
        self.cache[text_hash] = embedding
        
        # Save cache periodically (every 10 new entries)
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    def save(self) -> None:
        """Explicitly save cache to disk."""
        self._save_cache()


class SemanticAnalyzer:
    """
    Semantic analyzer for heading classification using sentence transformers.
    
    Provides Tier 2 analysis for ambiguous heading candidates using:
    - Semantic similarity scoring against known heading patterns
    - Context window analysis (±2 sentences) for coherence validation
    - Embedding cache for performance optimization
    - Lazy loading to activate only when rule-based confidence is 0.3-0.85
    """
    
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        """
        Initialize semantic analyzer.
        
        Args:
            cache_dir: Directory for embedding cache
        """
        self.cache = EmbeddingCache(cache_dir)
        self.confidence_threshold = (0.3, 0.85)  # Range for semantic analysis activation
        
        # Pre-defined heading patterns for similarity comparison
        self.heading_patterns = {
            'h1_patterns': [
                "Introduction", "Chapter 1", "Overview", "Background", "Abstract",
                "Executive Summary", "Methodology", "Results", "Discussion", 
                "Conclusion", "References", "Bibliography", "Appendix"
            ],
            'h2_patterns': [
                "Section 1.1", "Analysis", "Implementation", "Evaluation",
                "Related Work", "Literature Review", "Data Collection",
                "Experimental Setup", "Findings", "Recommendations"
            ],
            'h3_patterns': [
                "Subsection 1.1.1", "Definition", "Example", "Case Study",
                "Detailed Analysis", "Specific Implementation", "Technical Details",
                "Performance Metrics", "Validation Results", "Future Work"
            ]
        }
        
        # Semantic similarity thresholds
        self.similarity_thresholds = {
            'high_confidence': 0.7,    # Strong semantic match
            'medium_confidence': 0.5,  # Moderate semantic match
            'low_confidence': 0.3      # Weak semantic match
        }
        
        # Context window size (sentences before/after)
        self.context_window_size = 2
    
    def analyze_ambiguous_candidates(self, candidates: List[HeadingCandidate], 
                                   text_blocks: List[TextBlock]) -> List[HeadingCandidate]:
        """
        Analyze ambiguous heading candidates using semantic analysis.
        
        Args:
            candidates: List of heading candidates to analyze
            text_blocks: All text blocks from document for context
            
        Returns:
            List of candidates with updated confidence scores and semantic features
        """
        # Filter candidates that need semantic analysis
        ambiguous_candidates = self._filter_ambiguous_candidates(candidates)
        
        if not ambiguous_candidates:
            logger.info("No ambiguous candidates found for semantic analysis")
            return candidates
        
        logger.info(f"Performing semantic analysis on {len(ambiguous_candidates)} ambiguous candidates")
        
        # Build semantic context
        context = self._build_semantic_context(candidates, text_blocks)
        
        # Initialize heading pattern embeddings
        self._initialize_pattern_embeddings(context)
        
        # Analyze each ambiguous candidate
        updated_candidates = []
        for candidate in candidates:
            if candidate in ambiguous_candidates:
                updated_candidate = self._analyze_candidate_semantics(candidate, context)
                updated_candidates.append(updated_candidate)
            else:
                updated_candidates.append(candidate)
        
        # Save cache after analysis
        self.cache.save()
        
        logger.info("Semantic analysis completed")
        return updated_candidates
    
    def _filter_ambiguous_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """
        Filter candidates that need semantic analysis based on confidence thresholds.
        
        Args:
            candidates: List of all candidates
            
        Returns:
            List of candidates requiring semantic analysis
        """
        min_conf, max_conf = self.confidence_threshold
        ambiguous = [
            candidate for candidate in candidates
            if min_conf <= candidate.confidence_score < max_conf
        ]
        
        logger.debug(f"Found {len(ambiguous)} candidates in ambiguous range ({min_conf}-{max_conf})")
        return ambiguous
    
    def _build_semantic_context(self, candidates: List[HeadingCandidate], 
                              text_blocks: List[TextBlock]) -> SemanticContext:
        """
        Build semantic context for analysis.
        
        Args:
            candidates: List of heading candidates
            text_blocks: All document text blocks
            
        Returns:
            Semantic context with embeddings and context windows
        """
        # Extract context windows for each candidate
        candidate_contexts = {}
        
        for candidate in candidates:
            candidate_id = self._get_candidate_id(candidate)
            before_context, after_context = self._extract_context_window(
                candidate, text_blocks
            )
            candidate_contexts[candidate_id] = (before_context, after_context)
        
        return SemanticContext(
            document_text_blocks=text_blocks,
            candidate_contexts=candidate_contexts,
            document_embeddings={},
            heading_pattern_embeddings={}
        )
    
    def _initialize_pattern_embeddings(self, context: SemanticContext) -> None:
        """
        Initialize embeddings for heading patterns.
        
        Args:
            context: Semantic context to update with pattern embeddings
        """
        model, np = _get_sentence_transformer()
        
        logger.debug("Initializing heading pattern embeddings")
        
        for pattern_category, patterns in self.heading_patterns.items():
            for pattern in patterns:
                # Check cache first
                cached_embedding = self.cache.get_embedding(pattern)
                if cached_embedding is not None:
                    context.heading_pattern_embeddings[pattern] = cached_embedding
                else:
                    # Generate new embedding
                    embedding = model.encode([pattern])[0]
                    context.heading_pattern_embeddings[pattern] = embedding
                    self.cache.store_embedding(pattern, embedding)
        
        logger.debug(f"Initialized {len(context.heading_pattern_embeddings)} pattern embeddings")
    
    def _analyze_candidate_semantics(self, candidate: HeadingCandidate, 
                                   context: SemanticContext) -> HeadingCandidate:
        """
        Analyze semantic features of a candidate.
        
        Args:
            candidate: Candidate to analyze
            context: Semantic context
            
        Returns:
            Updated candidate with semantic analysis results
        """
        model, np = _get_sentence_transformer()
        
        # Get candidate text embedding
        candidate_embedding = self._get_text_embedding(candidate.text, context)
        
        # Calculate similarity scores against heading patterns
        pattern_similarities = self._calculate_pattern_similarities(
            candidate_embedding, context, np
        )
        
        # Analyze context coherence
        context_coherence = self._analyze_context_coherence(candidate, context)
        
        # Calculate semantic confidence score
        semantic_confidence = self._calculate_semantic_confidence(
            pattern_similarities, context_coherence
        )
        
        # Update candidate with semantic features
        updated_candidate = self._update_candidate_with_semantics(
            candidate, semantic_confidence, pattern_similarities, context_coherence
        )
        
        logger.debug(f"Semantic analysis for '{candidate.text[:30]}...': "
                    f"confidence={semantic_confidence:.3f}")
        
        return updated_candidate
    
    def _extract_context_window(self, candidate: HeadingCandidate, 
                              text_blocks: List[TextBlock]) -> Tuple[str, str]:
        """
        Extract context window (±2 sentences) around candidate.
        
        Args:
            candidate: Heading candidate
            text_blocks: All document text blocks
            
        Returns:
            Tuple of (before_context, after_context)
        """
        if not candidate.text_block:
            return "", ""
        
        # Find candidate position in text blocks
        candidate_index = -1
        for i, block in enumerate(text_blocks):
            if (block.page == candidate.text_block.page and 
                abs(block.bbox[1] - candidate.text_block.bbox[1]) < 1.0):
                candidate_index = i
                break
        
        if candidate_index == -1:
            return "", ""
        
        # Extract before context
        before_blocks = []
        for i in range(max(0, candidate_index - self.context_window_size), candidate_index):
            if not text_blocks[i].is_empty():
                before_blocks.append(text_blocks[i].text)
        
        # Extract after context
        after_blocks = []
        for i in range(candidate_index + 1, 
                      min(len(text_blocks), candidate_index + 1 + self.context_window_size)):
            if not text_blocks[i].is_empty():
                after_blocks.append(text_blocks[i].text)
        
        before_context = " ".join(before_blocks)
        after_context = " ".join(after_blocks)
        
        return before_context, after_context
    
    def _get_text_embedding(self, text: str, context: SemanticContext) -> Any:
        """
        Get embedding for text, using cache if available.
        
        Args:
            text: Text to embed
            context: Semantic context
            
        Returns:
            Text embedding
        """
        # Check cache first
        cached_embedding = self.cache.get_embedding(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate new embedding
        model, np = _get_sentence_transformer()
        embedding = model.encode([text])[0]
        
        # Store in cache
        self.cache.store_embedding(text, embedding)
        
        return embedding
    
    def _calculate_pattern_similarities(self, candidate_embedding: Any, 
                                      context: SemanticContext, np: Any) -> Dict[str, float]:
        """
        Calculate similarity scores against heading patterns.
        
        Args:
            candidate_embedding: Embedding of candidate text
            context: Semantic context with pattern embeddings
            np: NumPy module
            
        Returns:
            Dictionary of pattern similarities
        """
        similarities = {}
        
        for pattern, pattern_embedding in context.heading_pattern_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(candidate_embedding, pattern_embedding) / (
                np.linalg.norm(candidate_embedding) * np.linalg.norm(pattern_embedding)
            )
            similarities[pattern] = float(similarity)
        
        return similarities
    
    def _analyze_context_coherence(self, candidate: HeadingCandidate, 
                                 context: SemanticContext) -> Dict[str, float]:
        """
        Analyze semantic coherence with surrounding context.
        
        Args:
            candidate: Heading candidate
            context: Semantic context
            
        Returns:
            Dictionary of coherence metrics
        """
        candidate_id = self._get_candidate_id(candidate)
        before_context, after_context = context.candidate_contexts.get(
            candidate_id, ("", "")
        )
        
        if not before_context and not after_context:
            return {'coherence_score': 0.5}  # Neutral score
        
        model, np = _get_sentence_transformer()
        
        # Get embeddings for context
        candidate_embedding = self._get_text_embedding(candidate.text, context)
        
        coherence_scores = []
        
        if before_context:
            before_embedding = self._get_text_embedding(before_context, context)
            before_similarity = np.dot(candidate_embedding, before_embedding) / (
                np.linalg.norm(candidate_embedding) * np.linalg.norm(before_embedding)
            )
            coherence_scores.append(before_similarity)
        
        if after_context:
            after_embedding = self._get_text_embedding(after_context, context)
            after_similarity = np.dot(candidate_embedding, after_embedding) / (
                np.linalg.norm(candidate_embedding) * np.linalg.norm(after_embedding)
            )
            coherence_scores.append(after_similarity)
        
        # Calculate average coherence
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
        
        return {
            'coherence_score': float(avg_coherence),
            'context_length': len(before_context) + len(after_context),
            'has_before_context': bool(before_context),
            'has_after_context': bool(after_context)
        }
    
    def _calculate_semantic_confidence(self, pattern_similarities: Dict[str, float],
                                     context_coherence: Dict[str, float]) -> float:
        """
        Calculate overall semantic confidence score.
        
        Args:
            pattern_similarities: Similarity scores against heading patterns
            context_coherence: Context coherence metrics
            
        Returns:
            Semantic confidence score (0.0-1.0)
        """
        # Find best pattern matches for each heading level
        h1_similarities = [
            sim for pattern, sim in pattern_similarities.items()
            if any(h1_pattern in pattern for h1_pattern in self.heading_patterns['h1_patterns'])
        ]
        h2_similarities = [
            sim for pattern, sim in pattern_similarities.items()
            if any(h2_pattern in pattern for h2_pattern in self.heading_patterns['h2_patterns'])
        ]
        h3_similarities = [
            sim for pattern, sim in pattern_similarities.items()
            if any(h3_pattern in pattern for h3_pattern in self.heading_patterns['h3_patterns'])
        ]
        
        # Get maximum similarity for each level
        max_h1_sim = max(h1_similarities) if h1_similarities else 0.0
        max_h2_sim = max(h2_similarities) if h2_similarities else 0.0
        max_h3_sim = max(h3_similarities) if h3_similarities else 0.0
        
        # Overall pattern similarity (best match across all levels)
        pattern_confidence = max(max_h1_sim, max_h2_sim, max_h3_sim)
        
        # Context coherence contribution
        coherence_score = context_coherence.get('coherence_score', 0.5)
        
        # Combine pattern similarity (70%) and context coherence (30%)
        semantic_confidence = (pattern_confidence * 0.7) + (coherence_score * 0.3)
        
        return max(0.0, min(1.0, semantic_confidence))
    
    def _update_candidate_with_semantics(self, candidate: HeadingCandidate,
                                       semantic_confidence: float,
                                       pattern_similarities: Dict[str, float],
                                       context_coherence: Dict[str, float]) -> HeadingCandidate:
        """
        Update candidate with semantic analysis results.
        
        Args:
            candidate: Original candidate
            semantic_confidence: Calculated semantic confidence
            pattern_similarities: Pattern similarity scores
            context_coherence: Context coherence metrics
            
        Returns:
            Updated candidate with semantic features
        """
        # Create updated candidate
        updated_candidate = HeadingCandidate(
            text=candidate.text,
            page=candidate.page,
            confidence_score=candidate.confidence_score,
            formatting_features=candidate.formatting_features.copy(),
            level_indicators=candidate.level_indicators.copy(),
            text_block=candidate.text_block
        )
        
        # Add semantic features
        updated_candidate.formatting_features.update({
            'semantic_confidence': semantic_confidence,
            'pattern_similarities': pattern_similarities,
            'context_coherence': context_coherence,
            'semantic_analysis_applied': True
        })
        
        # Update overall confidence score (combine rule-based + semantic)
        # Use weighted average: rule-based (60%) + semantic (40%)
        combined_confidence = (
            candidate.confidence_score * 0.6 + 
            semantic_confidence * 0.4
        )
        updated_candidate.confidence_score = combined_confidence
        
        # Add semantic indicators
        if semantic_confidence >= self.similarity_thresholds['high_confidence']:
            updated_candidate.add_level_indicator('high_semantic_confidence')
        elif semantic_confidence >= self.similarity_thresholds['medium_confidence']:
            updated_candidate.add_level_indicator('medium_semantic_confidence')
        else:
            updated_candidate.add_level_indicator('low_semantic_confidence')
        
        # Add best pattern match indicator
        best_pattern = max(pattern_similarities.keys(), 
                          key=lambda k: pattern_similarities[k])
        updated_candidate.add_level_indicator(f'best_pattern_{best_pattern.replace(" ", "_")}')
        
        return updated_candidate
    
    def _get_candidate_id(self, candidate: HeadingCandidate) -> str:
        """Generate unique ID for candidate."""
        return f"{candidate.page}_{candidate.text[:20]}_{id(candidate)}"
    
    def get_semantic_features(self, candidate: HeadingCandidate) -> Dict[str, Any]:
        """
        Extract semantic features from analyzed candidate.
        
        Args:
            candidate: Analyzed candidate
            
        Returns:
            Dictionary of semantic features
        """
        features = candidate.formatting_features
        
        return {
            'semantic_confidence': features.get('semantic_confidence', 0.0),
            'pattern_similarities': features.get('pattern_similarities', {}),
            'context_coherence': features.get('context_coherence', {}),
            'has_semantic_analysis': features.get('semantic_analysis_applied', False),
            'best_pattern_similarity': max(
                features.get('pattern_similarities', {}).values(),
                default=0.0
            )
        }
    
    def analyze_semantic_distribution(self, candidates: List[HeadingCandidate]) -> Dict[str, Any]:
        """
        Analyze semantic confidence distribution across candidates.
        
        Args:
            candidates: List of analyzed candidates
            
        Returns:
            Dictionary with semantic analysis statistics
        """
        semantic_candidates = [
            c for c in candidates 
            if c.formatting_features.get('semantic_analysis_applied', False)
        ]
        
        if not semantic_candidates:
            return {'analyzed_count': 0}
        
        semantic_confidences = [
            c.formatting_features.get('semantic_confidence', 0.0)
            for c in semantic_candidates
        ]
        
        return {
            'analyzed_count': len(semantic_candidates),
            'total_candidates': len(candidates),
            'analysis_percentage': (len(semantic_candidates) / len(candidates)) * 100,
            'avg_semantic_confidence': sum(semantic_confidences) / len(semantic_confidences),
            'min_semantic_confidence': min(semantic_confidences),
            'max_semantic_confidence': max(semantic_confidences),
            'high_confidence_count': sum(
                1 for conf in semantic_confidences 
                if conf >= self.similarity_thresholds['high_confidence']
            ),
            'medium_confidence_count': sum(
                1 for conf in semantic_confidences 
                if self.similarity_thresholds['medium_confidence'] <= conf < self.similarity_thresholds['high_confidence']
            ),
            'low_confidence_count': sum(
                1 for conf in semantic_confidences 
                if conf < self.similarity_thresholds['medium_confidence']
            )
        }


def analyze_ambiguous_candidates(candidates: List[HeadingCandidate], 
                               text_blocks: List[TextBlock],
                               cache_dir: str = ".cache/embeddings") -> List[HeadingCandidate]:
    """
    Convenience function to analyze ambiguous candidates with semantic analysis.
    
    Args:
        candidates: List of heading candidates
        text_blocks: All document text blocks for context
        cache_dir: Directory for embedding cache
        
    Returns:
        List of candidates with semantic analysis applied
    """
    analyzer = SemanticAnalyzer(cache_dir)
    return analyzer.analyze_ambiguous_candidates(candidates, text_blocks)


def get_semantic_statistics(candidates: List[HeadingCandidate]) -> Dict[str, Any]:
    """
    Get semantic analysis statistics for candidates.
    
    Args:
        candidates: List of analyzed candidates
        
    Returns:
        Dictionary with semantic statistics
    """
    analyzer = SemanticAnalyzer()
    return analyzer.analyze_semantic_distribution(candidates)