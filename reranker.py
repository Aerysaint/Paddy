"""
Reranker module for improving retrieval results using cross-encoder models.

This module provides the Reranker class that uses cross-encoder models to rerank
candidate chunks from the initial retrieval step, providing more contextual
understanding of query-document relevance.

Usage:
    reranker = Reranker()
    reranked_chunks = reranker.rerank(query, candidate_chunks)
"""

import logging
from typing import List, Dict, Any, Tuple
import torch
from sentence_transformers import CrossEncoder
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder based reranker for improving retrieval results.
    
    Uses a cross-encoder model to score query-document pairs more accurately
    than bi-encoder similarity, providing better ranking of candidate chunks.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        
        logger.info(f"Initializing Reranker with model: {model_name}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Successfully loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
            raise
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = None) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunks using cross-encoder scores.
        
        Args:
            query: The search query string
            chunks: List of candidate chunks from initial retrieval
            top_n: Number of top results to return (None = return all)
            
        Returns:
            List of chunks reordered by cross-encoder relevance scores
        """
        if not chunks:
            logger.warning("No chunks provided for reranking")
            return []
        
        if not query.strip():
            logger.warning("Empty query provided for reranking")
            return chunks
        
        logger.info(f"Reranking {len(chunks)} chunks for query: '{query[:50]}...'")
        
        try:
            # Create query-chunk pairs for cross-encoder
            pairs = []
            for chunk in chunks:
                chunk_text = chunk.get('text_content', '')
                if chunk_text.strip():
                    pairs.append([query, chunk_text])
                else:
                    # Handle empty chunks by using heading text as fallback
                    fallback_text = chunk.get('heading_text', '')
                    pairs.append([query, fallback_text])
            
            if not pairs:
                logger.warning("No valid text content found in chunks")
                return chunks
            
            # Get cross-encoder scores
            logger.info("Computing cross-encoder relevance scores...")
            scores = self.model.predict(pairs)
            
            # Convert to numpy array if it isn't already
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)
            
            # Add reranker scores to chunks and sort
            reranked_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_copy = chunk.copy()
                chunk_copy['reranker_score'] = float(scores[i])
                reranked_chunks.append(chunk_copy)
            
            # Sort by reranker score (descending)
            reranked_chunks.sort(key=lambda x: x['reranker_score'], reverse=True)
            
            # Limit results if top_n specified
            if top_n is not None:
                reranked_chunks = reranked_chunks[:top_n]
            
            logger.info(f"Reranking complete. Top score: {reranked_chunks[0]['reranker_score']:.4f}")
            
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original chunks if reranking fails
            return chunks
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_type": "cross-encoder",
            "loaded": self.model is not None
        }


def main():
    """
    Example usage and testing of the Reranker class.
    """
    import argparse
    import json
    from retriever import Retriever, load_chunks_from_file
    
    parser = argparse.ArgumentParser(description="Rerank retrieval results using cross-encoder")
    parser.add_argument("chunks_file", help="JSON file containing chunks from chunker.py")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--initial-top-n", "-i", type=int, default=50, 
                       help="Number of initial results from bi-encoder")
    parser.add_argument("--final-top-n", "-f", type=int, default=10, 
                       help="Number of final results after reranking")
    parser.add_argument("--retriever-model", "-r", default="BAAI/bge-small-en-v1.5", 
                       help="Bi-encoder model for initial retrieval")
    parser.add_argument("--reranker-model", "-m", default="cross-encoder/ms-marco-MiniLM-L-6-v2", 
                       help="Cross-encoder model for reranking")
    
    args = parser.parse_args()
    
    # Load chunks
    chunks = load_chunks_from_file(args.chunks_file)
    if not chunks:
        print("No chunks loaded. Exiting.")
        return
    
    print(f"Loaded {len(chunks)} chunks from {args.chunks_file}")
    
    # Step 1: Initial retrieval with bi-encoder
    print(f"\nStep 1: Initial retrieval with {args.retriever_model}")
    retriever = Retriever(model_name=args.retriever_model)
    retriever.build_index(chunks)
    
    initial_results = retriever.search(args.query, top_n=args.initial_top_n)
    print(f"Retrieved {len(initial_results)} initial candidates")
    
    # Step 2: Rerank with cross-encoder
    print(f"\nStep 2: Reranking with {args.reranker_model}")
    reranker = Reranker(model_name=args.reranker_model)
    
    final_results = reranker.rerank(args.query, initial_results, top_n=args.final_top_n)
    print(f"Reranked to {len(final_results)} final results")
    
    # Display results
    print(f"\nQuery: '{args.query}'")
    print("=" * 80)
    
    for i, result in enumerate(final_results, 1):
        print(f"\n{i}. Reranker Score: {result['reranker_score']:.4f} | "
              f"Similarity Score: {result.get('similarity_score', 'N/A'):.4f}")
        print(f"   Source: {result['source_document']} (Page {result['page_number']})")
        print(f"   Level: {result['heading_level']} - {result['heading_text']}")
        print(f"   Content: {result['text_content'][:200]}...")
    
    # Show comparison with initial ranking
    print(f"\n\nComparison: Initial vs Reranked Order")
    print("-" * 50)
    
    # Find how the top results changed
    initial_top_5 = initial_results[:5]
    final_top_5 = final_results[:5]
    
    print("Initial Top 5 (by similarity):")
    for i, result in enumerate(initial_top_5, 1):
        print(f"  {i}. {result['heading_text'][:50]}... (sim: {result.get('similarity_score', 0):.3f})")
    
    print("\nFinal Top 5 (by reranker):")
    for i, result in enumerate(final_top_5, 1):
        print(f"  {i}. {result['heading_text'][:50]}... (rerank: {result['reranker_score']:.3f})")


if __name__ == "__main__":
    main()