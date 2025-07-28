"""
Semantic retrieval module for finding similar chunks based on query embeddings.

This module provides the Retriever class that can:
1. Load a bi-encoder model for generating embeddings
2. Build an index from chunks with their embeddings
3. Search for semantically similar chunks given a query

Usage:
    retriever = Retriever()
    retriever.build_index(chunks)
    results = retriever.search("What is agile testing?", top_n=10)
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Semantic retrieval system for finding similar document chunks.
    
    Uses a bi-encoder model to generate embeddings for chunks and queries,
    then performs similarity search to find the most relevant chunks.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the retriever with a bi-encoder model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.embeddings = None
        self.index_built = False
        
        logger.info(f"Initializing Retriever with model: {model_name}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build the search index from a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries containing 'text_content' and metadata
        """
        if not chunks:
            logger.warning("No chunks provided to build index")
            return
        
        logger.info(f"Building index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        
        # Extract text content for embedding
        texts = []
        for chunk in chunks:
            text_content = chunk.get('text_content', '')
            if not text_content:
                logger.warning(f"Empty text content in chunk: {chunk.get('heading_text', 'Unknown')}")
                texts.append("")
            else:
                texts.append(text_content)
        
        # Generate embeddings
        try:
            logger.info("Generating embeddings...")
            self.embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=True,
                batch_size=32
            )
            
            # Convert to numpy array for easier manipulation
            self.embeddings = np.array(self.embeddings)
            
            self.index_built = True
            logger.info(f"Index built successfully. Shape: {self.embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def search(self, query: str, top_n: int = 50) -> List[Dict[str, Any]]:
        """
        Search for the most similar chunks to the given query.
        
        Args:
            query: Text query to search for
            top_n: Number of top results to return
            
        Returns:
            List of dictionaries containing chunk data and similarity scores
        """
        if not self.index_built:
            logger.error("Index not built. Call build_index() first.")
            return []
        
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        logger.info(f"Searching for: '{query[:50]}...' (top {top_n} results)")
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            query_embedding = np.array(query_embedding)
            
            # Calculate cosine similarities
            similarities = self._cosine_similarity(query_embedding, self.embeddings)
            
            # Get top-k indices
            top_indices = np.argsort(similarities[0])[::-1][:top_n]
            
            # Prepare results
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(similarities[0][idx])
                results.append(chunk)
            
            logger.info(f"Found {len(results)} results. Top score: {results[0]['similarity_score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _cosine_similarity(self, query_embeddings: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and chunk embeddings.
        
        Args:
            query_embeddings: Query embedding matrix (1 x embedding_dim)
            chunk_embeddings: Chunk embedding matrix (n_chunks x embedding_dim)
            
        Returns:
            Similarity scores matrix (1 x n_chunks)
        """
        # Normalize embeddings
        query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        chunk_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.dot(query_norm, chunk_norm.T)
        
        return similarities
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.index_built:
            return {"status": "Index not built"}
        
        return {
            "status": "Index built",
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "total_text_length": sum(len(chunk.get('text_content', '')) for chunk in self.chunks),
            "source_documents": list(set(chunk.get('source_document', 'Unknown') for chunk in self.chunks))
        }
    
    def save_index(self, filepath: str) -> None:
        """
        Save the index to disk for later loading.
        
        Args:
            filepath: Path to save the index
        """
        if not self.index_built:
            logger.error("Cannot save index: not built yet")
            return
        
        try:
            index_data = {
                "model_name": self.model_name,
                "chunks": self.chunks,
                "embeddings": self.embeddings.tolist()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Index saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self, filepath: str) -> None:
        """
        Load a previously saved index from disk.
        
        Args:
            filepath: Path to the saved index file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Verify model compatibility
            saved_model = index_data.get("model_name")
            if saved_model != self.model_name:
                logger.warning(f"Model mismatch: current={self.model_name}, saved={saved_model}")
            
            self.chunks = index_data["chunks"]
            self.embeddings = np.array(index_data["embeddings"])
            self.index_built = True
            
            logger.info(f"Index loaded from: {filepath}")
            logger.info(f"Loaded {len(self.chunks)} chunks with embeddings shape: {self.embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise


def load_chunks_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSON file created by chunker.py.
    
    Args:
        filepath: Path to the chunks JSON file
        
    Returns:
        List of chunk dictionaries
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to load chunks from {filepath}: {e}")
        return []


def main():
    """
    Example usage of the Retriever class.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic retrieval for document chunks")
    parser.add_argument("chunks_file", help="JSON file containing chunks from chunker.py")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--top-n", "-n", type=int, default=10, help="Number of results to return")
    parser.add_argument("--model", "-m", default="BAAI/bge-small-en-v1.5", help="Model name to use")
    parser.add_argument("--save-index", help="Save the built index to this file")
    parser.add_argument("--load-index", help="Load index from this file instead of building")
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = Retriever(model_name=args.model)
    
    # Load or build index
    if args.load_index:
        retriever.load_index(args.load_index)
    else:
        # Load chunks and build index
        chunks = load_chunks_from_file(args.chunks_file)
        if not chunks:
            print("No chunks loaded. Exiting.")
            return
        
        retriever.build_index(chunks)
        
        # Save index if requested
        if args.save_index:
            retriever.save_index(args.save_index)
    
    # Print index statistics
    stats = retriever.get_stats()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Perform search if query provided
    if args.query:
        print(f"\nSearching for: '{args.query}'")
        print("-" * 50)
        
        results = retriever.search(args.query, top_n=args.top_n)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['similarity_score']:.4f}")
            print(f"   Source: {result['source_document']} (Page {result['page_number']})")
            print(f"   Level: {result['heading_level']} - {result['heading_text']}")
            print(f"   Content: {result['text_content'][:200]}...")
    else:
        print("\nIndex built successfully. Use --query to search.")


if __name__ == "__main__":
    main()