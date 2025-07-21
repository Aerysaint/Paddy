"""
Hybrid confidence fusion system for PDF outline extraction.

This module implements multi-tier confidence scoring that combines rule-based,
semantic, and visual scores with document type detection for adaptive weighting.
Includes tier-based decision making, early termination for high-confidence cases,
and performance optimizations to meet 10-second constraint for 50-page PDFs.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import Counter

from .data_models import HeadingCandidate, TextBlock
from .structure_analyzer import StructureAnalyzer
from .semantic_analyzer import SemanticAnalyzer
from .vlm_analyzer import VLMAnalyzer
from .logging_config import setup_logging

logger = setup_logging()


class DocumentType(Enum):
    """Document type classification for adaptive weighting."""
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    BUSINESS = "business"
    LEGAL = "legal"
    UNKNOWN = "unknown"


@dataclass
class ConfidenceWeights:
    """Confidence fusion weights for different document types."""
    rule_based: float
    semantic: float
    visual: float
    
    def normalize(self) -> 'ConfidenceWeights':
        """Normalize weights to sum to 1.0."""
        total = self.rule_based + self.semantic + self.visual
        if total == 0:
            return ConfidenceWeights(1.0, 0.0, 0.0)
        return ConfidenceWeights(
            self.rule_based / total,
            self.semantic / total,
            self.visual / total
        )


@dataclass
class FusionResult:
    """Result of confidence fusion analysis."""
    candidates: List[HeadingCandidate]
    document_type: DocumentType
    processing_stats: Dict[str, Any]
    performance_metrics: Dict[str, float]


@dataclass
class ProcessingStats:
    """Statistics for processing performance tracking."""
    total_candidates: int
    high_confidence_count: int
    semantic_analysis_count: int
    vlm_analysis_count: int
    early_termination_count: int
    processing_time: float
    tier_distribution: Dict[str, int]


class DocumentTypeDetector:
    """Detects document type for adaptive confidence weighting."""
    
    def __init__(self):
        """Initialize document type detector."""
        self.type_indicators = {
            DocumentType.ACADEMIC: {
                'keywords': {
                    'abstract', 'methodology', 'results', 'discussion', 'conclusion',
                    'references', 'bibliography', 'literature review', 'hypothesis',
                    'experiment', 'analysis', 'findings', 'research', 'study'
                },
                'patterns': [
                    r'\b(figure|table|equation)\s+\d+',
                    r'\b(section|chapter)\s+\d+',
                    r'\bcitation\b',
                    r'\bet\s+al\.',
                    r'\b\d{4}\b.*\b(journal|conference|proceedings)\b'
                ],
                'structure_indicators': ['numbered_sections', 'references_section']
            },
            DocumentType.TECHNICAL: {
                'keywords': {
                    'implementation', 'specification', 'requirements', 'architecture',
                    'design', 'system', 'component', 'interface', 'api', 'protocol',
                    'algorithm', 'procedure', 'configuration', 'installation', 'setup'
                },
                'patterns': [
                    r'\bversion\s+\d+\.\d+',
                    r'\bapi\b',
                    r'\bcode\b',
                    r'\bfunction\b',
                    r'\bclass\b',
                    r'\bmethod\b',
                    r'\bparameter\b'
                ],
                'structure_indicators': ['code_blocks', 'hierarchical_numbering']
            },
            DocumentType.BUSINESS: {
                'keywords': {
                    'executive summary', 'business', 'strategy', 'market', 'revenue',
                    'profit', 'budget', 'financial', 'investment', 'roi', 'kpi',
                    'objectives', 'goals', 'recommendations', 'proposal', 'plan'
                },
                'patterns': [
                    r'\$\d+',
                    r'\b\d+%\b',
                    r'\bq[1-4]\b',
                    r'\bfiscal\s+year\b',
                    r'\bquarter\b'
                ],
                'structure_indicators': ['executive_summary', 'financial_sections']
            },
            DocumentType.LEGAL: {
                'keywords': {
                    'article', 'section', 'clause', 'paragraph', 'subsection',
                    'whereas', 'therefore', 'hereby', 'pursuant', 'compliance',
                    'regulation', 'statute', 'law', 'legal', 'contract', 'agreement'
                },
                'patterns': [
                    r'\barticle\s+\d+',
                    r'\bsection\s+\d+',
                    r'\bclause\s+\d+',
                    r'\b\(\w+\)\s*\(',
                    r'\bwhereas\b',
                    r'\btherefore\b'
                ],
                'structure_indicators': ['numbered_articles', 'legal_clauses']
            }
        }
    
    def detect_document_type(self, text_blocks: List[TextBlock]) -> DocumentType:
        """
        Detect document type based on content analysis.
        
        Args:
            text_blocks: List of text blocks from document
            
        Returns:
            Detected document type
        """
        if not text_blocks:
            return DocumentType.UNKNOWN
        
        # Combine all text for analysis
        full_text = ' '.join(block.text.lower() for block in text_blocks if not block.is_empty())
        
        # Calculate scores for each document type
        type_scores = {}
        
        for doc_type, indicators in self.type_indicators.items():
            score = self._calculate_type_score(full_text, text_blocks, indicators)
            type_scores[doc_type] = score
        
        # Find the type with highest score
        best_type = max(type_scores.keys(), key=lambda t: type_scores[t])
        best_score = type_scores[best_type]
        
        # Require minimum confidence threshold
        if best_score < 0.3:
            logger.debug(f"Document type detection confidence too low: {best_score:.3f}")
            return DocumentType.UNKNOWN
        
        logger.info(f"Detected document type: {best_type.value} (confidence: {best_score:.3f})")
        return best_type
    
    def _calculate_type_score(self, full_text: str, text_blocks: List[TextBlock],
                            indicators: Dict[str, Any]) -> float:
        """Calculate type score based on indicators."""
        score = 0.0
        
        # Keyword matching (40% of score)
        keyword_score = self._score_keywords(full_text, indicators['keywords'])
        score += keyword_score * 0.4
        
        # Pattern matching (35% of score)
        pattern_score = self._score_patterns(full_text, indicators['patterns'])
        score += pattern_score * 0.35
        
        # Structure indicators (25% of score)
        structure_score = self._score_structure(text_blocks, indicators['structure_indicators'])
        score += structure_score * 0.25
        
        return score
    
    def _score_keywords(self, text: str, keywords: Set[str]) -> float:
        """Score based on keyword presence."""
        if not keywords:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword in text)
        return min(1.0, matches / len(keywords) * 3)  # Scale up for better discrimination
    
    def _score_patterns(self, text: str, patterns: List[str]) -> float:
        """Score based on regex pattern matches."""
        import re
        
        if not patterns:
            return 0.0
        
        matches = 0
        for pattern in patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    matches += 1
            except re.error:
                continue
        
        return min(1.0, matches / len(patterns) * 2)  # Scale up for better discrimination
    
    def _score_structure(self, text_blocks: List[TextBlock], indicators: List[str]) -> float:
        """Score based on document structure indicators."""
        score = 0.0
        
        # Check for specific structural patterns
        if 'numbered_sections' in indicators:
            numbered_count = sum(1 for block in text_blocks 
                               if self._has_section_numbering(block.text))
            if numbered_count > 2:
                score += 0.5
        
        if 'references_section' in indicators:
            has_references = any('reference' in block.text.lower() 
                               for block in text_blocks)
            if has_references:
                score += 0.3
        
        if 'executive_summary' in indicators:
            has_exec_summary = any('executive summary' in block.text.lower() 
                                 for block in text_blocks)
            if has_exec_summary:
                score += 0.4
        
        return min(1.0, score)
    
    def _has_section_numbering(self, text: str) -> bool:
        """Check if text has section numbering pattern."""
        import re
        patterns = [
            r'^\d+\.',
            r'^\d+\.\d+',
            r'^section\s+\d+',
            r'^chapter\s+\d+'
        ]
        
        text_start = text.strip().lower()
        return any(re.match(pattern, text_start) for pattern in patterns)


class ConfidenceFusionSystem:
    """
    Hybrid confidence fusion system combining multiple analysis tiers.
    
    Implements:
    - Multi-tier confidence scoring (rule-based, semantic, visual)
    - Document type detection for adaptive weighting
    - Tier-based decision making with early termination
    - Performance optimizations for 10-second constraint
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize confidence fusion system.
        
        Args:
            cache_dir: Directory for caching models and embeddings
        """
        self.cache_dir = cache_dir
        
        # Initialize analyzers (lazy loading)
        self.structure_analyzer = None
        self.semantic_analyzer = None
        self.vlm_analyzer = None
        self.document_type_detector = DocumentTypeDetector()
        
        # Confidence thresholds for tier-based decision making
        self.confidence_thresholds = {
            'high_confidence': 0.85,    # Early termination threshold
            'semantic_range': (0.3, 0.85),  # Range for semantic analysis
            'vlm_threshold': 0.5        # Threshold for VLM analysis
        }
        
        # Document type specific weights
        self.type_weights = {
            DocumentType.ACADEMIC: ConfidenceWeights(0.40, 0.45, 0.15),
            DocumentType.TECHNICAL: ConfidenceWeights(0.50, 0.25, 0.25),
            DocumentType.BUSINESS: ConfidenceWeights(0.45, 0.35, 0.20),
            DocumentType.LEGAL: ConfidenceWeights(0.60, 0.25, 0.15),
            DocumentType.UNKNOWN: ConfidenceWeights(0.50, 0.30, 0.20)
        }
        
        # Performance tracking
        self.performance_target = 10.0  # 10 second target for 50-page PDFs
        self.start_time = None
    
    def analyze_document_with_fusion(self, text_blocks: List[TextBlock]) -> FusionResult:
        """
        Analyze document using hybrid confidence fusion system.
        
        Args:
            text_blocks: List of text blocks from PDF
            
        Returns:
            Fusion result with analyzed candidates and metadata
        """
        self.start_time = time.time()
        logger.info(f"Starting hybrid confidence fusion analysis for {len(text_blocks)} text blocks")
        
        # Detect document type for adaptive weighting
        document_type = self.document_type_detector.detect_document_type(text_blocks)
        weights = self.type_weights[document_type].normalize()
        
        logger.info(f"Document type: {document_type.value}, "
                   f"weights: rule={weights.rule_based:.2f}, "
                   f"semantic={weights.semantic:.2f}, visual={weights.visual:.2f}")
        
        # Initialize processing statistics
        stats = ProcessingStats(
            total_candidates=0,
            high_confidence_count=0,
            semantic_analysis_count=0,
            vlm_analysis_count=0,
            early_termination_count=0,
            processing_time=0.0,
            tier_distribution={'tier1': 0, 'tier2': 0, 'tier3': 0}
        )
        
        # Tier 1: Rule-based analysis (fast filter)
        candidates = self._tier1_rule_based_analysis(text_blocks, stats)
        
        # Check performance constraint
        if self._check_performance_constraint():
            logger.warning("Performance constraint exceeded, skipping advanced tiers")
            return self._create_fusion_result(candidates, document_type, stats)
        
        # Tier 2: Semantic analysis for ambiguous cases
        candidates = self._tier2_semantic_analysis(candidates, text_blocks, weights, stats)
        
        # Check performance constraint again
        if self._check_performance_constraint():
            logger.warning("Performance constraint exceeded, skipping VLM tier")
            return self._create_fusion_result(candidates, document_type, stats)
        
        # Tier 3: VLM analysis for highest ambiguity cases
        candidates = self._tier3_vlm_analysis(candidates, text_blocks, weights, stats)
        
        # Final confidence fusion
        final_candidates = self._apply_final_fusion(candidates, weights, document_type)
        
        # Update final statistics
        stats.processing_time = time.time() - self.start_time
        stats.total_candidates = len(final_candidates)
        
        logger.info(f"Hybrid fusion analysis completed in {stats.processing_time:.2f}s")
        
        return self._create_fusion_result(final_candidates, document_type, stats)
    
    def _tier1_rule_based_analysis(self, text_blocks: List[TextBlock],
                                 stats: ProcessingStats) -> List[HeadingCandidate]:
        """
        Tier 1: Rule-based fast filter analysis.
        
        Args:
            text_blocks: Document text blocks
            stats: Processing statistics to update
            
        Returns:
            List of heading candidates with rule-based scores
        """
        logger.debug("Starting Tier 1: Rule-based analysis")
        
        # Initialize structure analyzer if needed
        if self.structure_analyzer is None:
            self.structure_analyzer = StructureAnalyzer()
        
        # Analyze document structure
        candidates, context = self.structure_analyzer.analyze_document_structure(text_blocks)
        
        # Count high-confidence candidates (early termination eligible)
        high_conf_candidates = [
            c for c in candidates 
            if c.confidence_score >= self.confidence_thresholds['high_confidence']
        ]
        
        stats.high_confidence_count = len(high_conf_candidates)
        stats.early_termination_count = stats.high_confidence_count
        stats.tier_distribution['tier1'] = len(candidates)
        
        logger.debug(f"Tier 1 completed: {len(candidates)} candidates, "
                    f"{len(high_conf_candidates)} high-confidence")
        
        return candidates
    
    def _tier2_semantic_analysis(self, candidates: List[HeadingCandidate],
                               text_blocks: List[TextBlock],
                               weights: ConfidenceWeights,
                               stats: ProcessingStats) -> List[HeadingCandidate]:
        """
        Tier 2: Semantic analysis for ambiguous cases.
        
        Args:
            candidates: Candidates from Tier 1
            text_blocks: Document text blocks
            weights: Document type weights
            stats: Processing statistics to update
            
        Returns:
            List of candidates with semantic analysis applied
        """
        # Skip semantic analysis if weight is very low
        if weights.semantic < 0.1:
            logger.debug("Skipping Tier 2: Semantic weight too low")
            return candidates
        
        logger.debug("Starting Tier 2: Semantic analysis")
        
        # Filter candidates needing semantic analysis
        min_conf, max_conf = self.confidence_thresholds['semantic_range']
        ambiguous_candidates = [
            c for c in candidates
            if min_conf <= c.confidence_score < max_conf
        ]
        
        if not ambiguous_candidates:
            logger.debug("No candidates need semantic analysis")
            return candidates
        
        # Initialize semantic analyzer if needed
        if self.semantic_analyzer is None:
            self.semantic_analyzer = SemanticAnalyzer(f"{self.cache_dir}/embeddings")
        
        # Apply semantic analysis
        try:
            analyzed_candidates = self.semantic_analyzer.analyze_ambiguous_candidates(
                candidates, text_blocks
            )
            
            stats.semantic_analysis_count = len(ambiguous_candidates)
            stats.tier_distribution['tier2'] = len(ambiguous_candidates)
            
            logger.debug(f"Tier 2 completed: {len(ambiguous_candidates)} candidates analyzed")
            return analyzed_candidates
            
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}, continuing with rule-based scores")
            return candidates
    
    def _tier3_vlm_analysis(self, candidates: List[HeadingCandidate],
                          text_blocks: List[TextBlock],
                          weights: ConfidenceWeights,
                          stats: ProcessingStats) -> List[HeadingCandidate]:
        """
        Tier 3: VLM analysis for highest ambiguity cases.
        
        Args:
            candidates: Candidates from Tier 2
            text_blocks: Document text blocks
            weights: Document type weights
            stats: Processing statistics to update
            
        Returns:
            List of candidates with VLM analysis applied
        """
        # Skip VLM analysis if weight is very low or performance constraint
        if weights.visual < 0.1:
            logger.debug("Skipping Tier 3: Visual weight too low")
            return candidates
        
        if self._check_performance_constraint(threshold=0.8):  # 80% of time budget
            logger.debug("Skipping Tier 3: Performance constraint")
            return candidates
        
        logger.debug("Starting Tier 3: VLM analysis")
        
        # Filter candidates needing VLM analysis
        high_ambiguity_candidates = [
            c for c in candidates
            if c.confidence_score < self.confidence_thresholds['vlm_threshold']
        ]
        
        if not high_ambiguity_candidates:
            logger.debug("No candidates need VLM analysis")
            return candidates
        
        # Limit VLM analysis to most ambiguous cases for performance
        max_vlm_candidates = min(len(high_ambiguity_candidates), 10)  # Limit to 10 candidates
        high_ambiguity_candidates = sorted(
            high_ambiguity_candidates, 
            key=lambda c: c.confidence_score
        )[:max_vlm_candidates]
        
        # Initialize VLM analyzer if needed
        if self.vlm_analyzer is None:
            self.vlm_analyzer = VLMAnalyzer(f"{self.cache_dir}/vlm_models")
        
        # Apply VLM analysis
        try:
            analyzed_candidates = self.vlm_analyzer.analyze_highest_ambiguity_candidates(
                candidates, text_blocks
            )
            
            stats.vlm_analysis_count = len(high_ambiguity_candidates)
            stats.tier_distribution['tier3'] = len(high_ambiguity_candidates)
            
            logger.debug(f"Tier 3 completed: {len(high_ambiguity_candidates)} candidates analyzed")
            return analyzed_candidates
            
        except Exception as e:
            logger.warning(f"VLM analysis failed: {e}, continuing without VLM scores")
            return candidates
    
    def _apply_final_fusion(self, candidates: List[HeadingCandidate],
                          weights: ConfidenceWeights,
                          document_type: DocumentType) -> List[HeadingCandidate]:
        """
        Apply final confidence fusion combining all tier scores.
        
        Args:
            candidates: Candidates with all available analysis
            weights: Document type weights
            document_type: Detected document type
            
        Returns:
            List of candidates with final fused confidence scores
        """
        logger.debug("Applying final confidence fusion")
        
        fused_candidates = []
        
        for candidate in candidates:
            # Extract scores from different tiers
            rule_score = self._extract_rule_based_score(candidate)
            semantic_score = self._extract_semantic_score(candidate)
            visual_score = self._extract_visual_score(candidate)
            
            # Apply weighted fusion
            fused_score = (
                rule_score * weights.rule_based +
                semantic_score * weights.semantic +
                visual_score * weights.visual
            )
            
            # Apply document type specific adjustments
            fused_score = self._apply_document_type_adjustments(
                fused_score, candidate, document_type
            )
            
            # Create updated candidate with fused score
            updated_candidate = self._create_fused_candidate(
                candidate, fused_score, rule_score, semantic_score, visual_score
            )
            
            fused_candidates.append(updated_candidate)
        
        logger.debug(f"Final fusion completed for {len(fused_candidates)} candidates")
        return fused_candidates
    
    def _extract_rule_based_score(self, candidate: HeadingCandidate) -> float:
        """Extract rule-based confidence score."""
        # Use the original confidence score as rule-based score
        return candidate.confidence_score
    
    def _extract_semantic_score(self, candidate: HeadingCandidate) -> float:
        """Extract semantic confidence score."""
        return candidate.formatting_features.get('semantic_confidence', 0.5)
    
    def _extract_visual_score(self, candidate: HeadingCandidate) -> float:
        """Extract visual/VLM confidence score."""
        return candidate.formatting_features.get('vlm_confidence', 0.5)
    
    def _apply_document_type_adjustments(self, base_score: float,
                                       candidate: HeadingCandidate,
                                       document_type: DocumentType) -> float:
        """Apply document type specific score adjustments."""
        adjusted_score = base_score
        
        # Academic document adjustments
        if document_type == DocumentType.ACADEMIC:
            # Boost numbered sections and references
            if 'numbering_level_' in str(candidate.level_indicators):
                adjusted_score += 0.1
            if any(keyword in candidate.text.lower() 
                  for keyword in ['reference', 'bibliography', 'abstract']):
                adjusted_score += 0.15
        
        # Technical document adjustments
        elif document_type == DocumentType.TECHNICAL:
            # Boost hierarchical numbering and technical terms
            if candidate.formatting_features.get('numbering_pattern') == 'hierarchical_decimal':
                adjusted_score += 0.15
            if any(keyword in candidate.text.lower() 
                  for keyword in ['implementation', 'specification', 'api']):
                adjusted_score += 0.1
        
        # Business document adjustments
        elif document_type == DocumentType.BUSINESS:
            # Boost executive summary and financial sections
            if any(keyword in candidate.text.lower() 
                  for keyword in ['executive', 'summary', 'financial', 'budget']):
                adjusted_score += 0.2
        
        # Legal document adjustments
        elif document_type == DocumentType.LEGAL:
            # Boost articles, sections, and clauses
            if any(keyword in candidate.text.lower() 
                  for keyword in ['article', 'section', 'clause']):
                adjusted_score += 0.15
            if candidate.formatting_features.get('numbering_pattern'):
                adjusted_score += 0.1
        
        return min(1.0, adjusted_score)
    
    def _create_fused_candidate(self, original: HeadingCandidate,
                              fused_score: float,
                              rule_score: float,
                              semantic_score: float,
                              visual_score: float) -> HeadingCandidate:
        """Create candidate with fused confidence score."""
        # Create new candidate with updated score
        fused_candidate = HeadingCandidate(
            text=original.text,
            page=original.page,
            confidence_score=fused_score,
            formatting_features=original.formatting_features.copy(),
            level_indicators=original.level_indicators.copy(),
            text_block=original.text_block
        )
        
        # Add fusion metadata
        fused_candidate.formatting_features.update({
            'fusion_applied': True,
            'rule_based_score': rule_score,
            'semantic_score': semantic_score,
            'visual_score': visual_score,
            'final_fused_score': fused_score
        })
        
        # Add fusion confidence indicator
        if fused_score >= self.confidence_thresholds['high_confidence']:
            fused_candidate.add_level_indicator('high_fusion_confidence')
        elif fused_score >= 0.6:
            fused_candidate.add_level_indicator('medium_fusion_confidence')
        else:
            fused_candidate.add_level_indicator('low_fusion_confidence')
        
        return fused_candidate
    
    def _check_performance_constraint(self, threshold: float = 1.0) -> bool:
        """
        Check if performance constraint is being approached.
        
        Args:
            threshold: Fraction of time budget (0.0-1.0)
            
        Returns:
            True if constraint is exceeded
        """
        if self.start_time is None:
            return False
        
        elapsed_time = time.time() - self.start_time
        time_budget = self.performance_target * threshold
        
        return elapsed_time > time_budget
    
    def _create_fusion_result(self, candidates: List[HeadingCandidate],
                            document_type: DocumentType,
                            stats: ProcessingStats) -> FusionResult:
        """Create fusion result with metadata."""
        # Calculate performance metrics
        processing_time = time.time() - self.start_time if self.start_time else 0.0
        
        performance_metrics = {
            'total_processing_time': processing_time,
            'time_per_candidate': processing_time / max(len(candidates), 1),
            'performance_target': self.performance_target,
            'within_target': processing_time <= self.performance_target,
            'efficiency_ratio': self.performance_target / max(processing_time, 0.1)
        }
        
        # Calculate processing statistics
        processing_stats = {
            'total_candidates': len(candidates),
            'high_confidence_count': len([c for c in candidates 
                                        if c.confidence_score >= self.confidence_thresholds['high_confidence']]),
            'semantic_analysis_count': stats.semantic_analysis_count,
            'vlm_analysis_count': stats.vlm_analysis_count,
            'early_termination_count': stats.early_termination_count,
            'tier_distribution': stats.tier_distribution,
            'confidence_distribution': self._calculate_confidence_distribution(candidates),
            'document_type': document_type.value
        }
        
        return FusionResult(
            candidates=candidates,
            document_type=document_type,
            processing_stats=processing_stats,
            performance_metrics=performance_metrics
        )
    
    def _calculate_confidence_distribution(self, candidates: List[HeadingCandidate]) -> Dict[str, int]:
        """Calculate distribution of confidence scores."""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for candidate in candidates:
            score = candidate.confidence_score
            if score >= self.confidence_thresholds['high_confidence']:
                distribution['high'] += 1
            elif score >= 0.6:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def get_fusion_statistics(self, result: FusionResult) -> Dict[str, Any]:
        """
        Get detailed fusion statistics.
        
        Args:
            result: Fusion result to analyze
            
        Returns:
            Dictionary with detailed statistics
        """
        stats = {
            'document_analysis': {
                'document_type': result.document_type.value,
                'total_candidates': len(result.candidates),
                'processing_time': result.performance_metrics['total_processing_time'],
                'within_performance_target': result.performance_metrics['within_target']
            },
            'tier_analysis': {
                'tier1_rule_based': result.processing_stats['tier_distribution']['tier1'],
                'tier2_semantic': result.processing_stats['tier_distribution']['tier2'],
                'tier3_vlm': result.processing_stats['tier_distribution']['tier3'],
                'early_termination_count': result.processing_stats['early_termination_count']
            },
            'confidence_analysis': {
                'high_confidence': result.processing_stats['confidence_distribution']['high'],
                'medium_confidence': result.processing_stats['confidence_distribution']['medium'],
                'low_confidence': result.processing_stats['confidence_distribution']['low']
            },
            'performance_metrics': result.performance_metrics,
            'fusion_effectiveness': {
                'semantic_analysis_rate': (
                    result.processing_stats['semantic_analysis_count'] / 
                    max(result.processing_stats['total_candidates'], 1)
                ),
                'vlm_analysis_rate': (
                    result.processing_stats['vlm_analysis_count'] / 
                    max(result.processing_stats['total_candidates'], 1)
                ),
                'early_termination_rate': (
                    result.processing_stats['early_termination_count'] / 
                    max(result.processing_stats['total_candidates'], 1)
                )
            }
        }
        
        return stats


def analyze_document_with_fusion(text_blocks: List[TextBlock],
                               cache_dir: str = ".cache") -> FusionResult:
    """
    Convenience function to analyze document with hybrid confidence fusion.
    
    Args:
        text_blocks: List of text blocks from PDF
        cache_dir: Directory for caching models and embeddings
        
    Returns:
        Fusion result with analyzed candidates and metadata
    """
    fusion_system = ConfidenceFusionSystem(cache_dir)
    return fusion_system.analyze_document_with_fusion(text_blocks)


def get_high_confidence_headings(text_blocks: List[TextBlock],
                               confidence_threshold: float = 0.85,
                               cache_dir: str = ".cache") -> List[HeadingCandidate]:
    """
    Get high-confidence heading candidates using fusion system.
    
    Args:
        text_blocks: List of text blocks from PDF
        confidence_threshold: Minimum confidence threshold
        cache_dir: Directory for caching
        
    Returns:
        List of high-confidence heading candidates
    """
    result = analyze_document_with_fusion(text_blocks, cache_dir)
    
    return [
        candidate for candidate in result.candidates
        if candidate.confidence_score >= confidence_threshold
    ]