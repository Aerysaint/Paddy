"""
Enhanced PDF processing pipeline with integrated retrieval and reranking.

This script processes PDFs and performs persona-based document search:
1. Extract headings using heading_extractor.py
2. Create chunks using chunker.py
3. Build master index from all chunks
4. Perform persona-based retrieval and reranking
5. Output ranked results in JSON format

Usage:
    python main.py <pdf_folder> --persona <persona_file> --job-to-be-done <job_description> [options]
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Import retrieval and reranking components
from retriever import Retriever
from reranker import Reranker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedPDFProcessor:
    """
    Enhanced PDF processor with integrated retrieval and reranking capabilities.
    """
    
    def __init__(
        self, 
        pdf_folder: str, 
        persona_file: Optional[str] = None,
        job_to_be_done: Optional[str] = None,
        output_dir: str = "chunks", 
        keep_intermediates: bool = False,
        retriever_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_candidates: int = 50,
        final_results: int = 20
    ):
        """
        Initialize the enhanced PDF processor.
        
        Args:
            pdf_folder: Path to folder containing PDF files
            persona_file: Path to persona JSON file
            job_to_be_done: Job description string
            output_dir: Directory to store output chunks
            keep_intermediates: Whether to keep intermediate JSON files
            retriever_model: Bi-encoder model for initial retrieval
            reranker_model: Cross-encoder model for reranking
            initial_candidates: Number of initial candidates to retrieve
            final_results: Number of final results to return
        """
        self.pdf_folder = Path(pdf_folder)
        self.persona_file = persona_file
        self.job_to_be_done = job_to_be_done
        self.output_dir = Path(output_dir)
        self.keep_intermediates = keep_intermediates
        self.retriever_model = retriever_model
        self.reranker_model = reranker_model
        self.initial_candidates = initial_candidates
        self.final_results = final_results
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        self.headings_dir = self.output_dir / "headings"
        self.chunks_dir = self.output_dir / "chunks"
        
        if keep_intermediates:
            self.headings_dir.mkdir(exist_ok=True)
        
        self.chunks_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.processed_files = []
        self.failed_files = []
        self.processing_stats = {}
        
        # Initialize retrieval components
        self.retriever = None
        self.reranker = None
        self.master_chunks = []
        self.persona_data = None
        
        # Challenge mode attributes
        self.challenge_mode = False
        self.challenge_data = None
        self.pdf_base_path = None
    
    def find_pdf_files(self) -> List[Path]:
        """
        Find PDF files - either from challenge input or folder.
        
        Returns:
            List of PDF file paths
        """
        if self.challenge_mode:
            return self.get_challenge_pdf_files()
        
        if not self.pdf_folder.exists():
            logger.error(f"PDF folder does not exist: {self.pdf_folder}")
            return []
        
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_folder}")
        
        return pdf_files
    
    def extract_headings(self, pdf_path: Path) -> Optional[Path]:
        """
        Extract headings from a PDF file using heading_extractor.py.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the generated headings JSON file, or None if failed
        """
        pdf_name = pdf_path.stem
        headings_file = self.headings_dir / f"{pdf_name}_headings.json" if self.keep_intermediates else Path(f"temp_{pdf_name}_headings.json")
        
        logger.info(f"Extracting headings from: {pdf_path.name}")
        
        try:
            # Run heading extractor
            cmd = [
                sys.executable, "heading_extractor.py",
                str(pdf_path),
                "-o", str(headings_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Heading extraction failed for {pdf_path.name}")
                logger.error(f"Error: {result.stderr}")
                return None
            
            if not headings_file.exists():
                logger.error(f"Headings file not created: {headings_file}")
                return None
            
            # Verify the JSON is valid
            try:
                with open(headings_file, 'r', encoding='utf-8') as f:
                    headings_data = json.load(f)
                
                if not headings_data.get('outline'):
                    logger.warning(f"No headings found in {pdf_path.name}")
                    return headings_file  # Still return it, chunker can handle empty outlines
                
                logger.info(f"Extracted {len(headings_data['outline'])} headings from {pdf_path.name}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in headings file {headings_file}: {e}")
                return None
            
            return headings_file
            
        except subprocess.TimeoutExpired:
            logger.error(f"Heading extraction timed out for {pdf_path.name}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during heading extraction for {pdf_path.name}: {e}")
            return None
    
    def create_chunks(self, headings_file: Path, pdf_path: Path) -> Optional[Path]:
        """
        Create chunks from headings and PDF using chunker.py.
        
        Args:
            headings_file: Path to the headings JSON file
            pdf_path: Path to the original PDF file
            
        Returns:
            Path to the generated chunks JSON file, or None if failed
        """
        pdf_name = pdf_path.stem
        chunks_file = self.chunks_dir / f"{pdf_name}_chunks.json"
        
        logger.info(f"Creating chunks for: {pdf_path.name}")
        
        try:
            # Run chunker
            cmd = [
                sys.executable, "chunker.py",
                str(headings_file),
                str(pdf_path),
                "-o", str(chunks_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Chunking failed for {pdf_path.name}")
                logger.error(f"Error: {result.stderr}")
                return None
            
            if not chunks_file.exists():
                logger.error(f"Chunks file not created: {chunks_file}")
                return None
            
            # Verify the chunks JSON is valid
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                
                if not isinstance(chunks_data, list):
                    logger.error(f"Invalid chunks format in {chunks_file}")
                    return None
                
                logger.info(f"Created {len(chunks_data)} chunks for {pdf_path.name}")
                
                # Calculate some statistics
                total_text_length = sum(len(chunk.get('text_content', '')) for chunk in chunks_data)
                avg_chunk_size = total_text_length / len(chunks_data) if chunks_data else 0
                
                self.processing_stats[pdf_name] = {
                    'num_chunks': len(chunks_data),
                    'total_text_length': total_text_length,
                    'avg_chunk_size': avg_chunk_size,
                    'source_file': str(pdf_path)
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in chunks file {chunks_file}: {e}")
                return None
            
            return chunks_file
            
        except subprocess.TimeoutExpired:
            logger.error(f"Chunking timed out for {pdf_path.name}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during chunking for {pdf_path.name}: {e}")
            return None
    # In main.py, inside the EnhancedPDFProcessor class

    def _enforce_document_diversity(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Ensures at least one result from every processed PDF is included.

        Args:
            results: The current list of top reranked chunks.
            query: The original search query.

        Returns:
            An updated list of results with missing documents represented.
        """
        logger.info("Enforcing document diversity in final results...")
        
        # Get the set of documents already in the results
        docs_in_results = set(Path(res['source_document']).name for res in results)
        
        # Get the set of all documents that were successfully processed
        all_processed_docs = set(Path(p['source_file']).name for p in self.processing_stats.values())
        
        missing_docs = all_processed_docs - docs_in_results
        
        if not missing_docs:
            logger.info("All documents are already represented. No changes needed.")
            return results

        logger.info(f"Missing documents to add: {', '.join(missing_docs)}")
        
        chunks_to_add = []
        for doc_name in missing_docs:
            # Find all chunks from this specific missing document
            chunks_from_doc = [
                chunk for chunk in self.master_chunks 
                if Path(chunk['source_document']).name == doc_name
            ]
            
            if not chunks_from_doc:
                continue

            # Find the single best chunk from this document by reranking its contents
            logger.info(f"Finding best chunk from '{doc_name}'...")
            best_chunk_for_doc = self.reranker.rerank(query, chunks_from_doc, top_n=1)
            
            if best_chunk_for_doc:
                logger.info(f"  -> Adding '{best_chunk_for_doc[0]['heading_text'][:50]}...'")
                chunks_to_add.append(best_chunk_for_doc[0])

        # Combine the original results with the best chunks from missing docs
        combined_results = results + chunks_to_add
        
        # Perform a final re-sort based on the reranker_score to ensure order is correct
        combined_results.sort(key=lambda x: x['reranker_score'], reverse=True)
        
        # Re-assign the importance_rank
        for i, result in enumerate(combined_results, 1):
            result['importance_rank'] = i
            
        logger.info(f"Diversity enforced. Final result count: {len(combined_results)}")
        return combined_results
    def load_persona(self) -> bool:
        """
        Load persona data from JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.persona_file:
            logger.warning("No persona file provided")
            return True  # Not required
        
        try:
            persona_path = Path(self.persona_file)
            if not persona_path.exists():
                logger.error(f"Persona file not found: {self.persona_file}")
                return False
            
            with open(persona_path, 'r', encoding='utf-8') as f:
                self.persona_data = json.load(f)
            
            logger.info(f"Loaded persona data from: {self.persona_file}")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in persona file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading persona file: {e}")
            return False
    
    def load_challenge_input(self, input_file: str) -> bool:
        """
        Load challenge input format from JSON file.
        
        Args:
            input_file: Path to challenge input JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            input_path = Path(input_file)
            if not input_path.exists():
                logger.error(f"Challenge input file not found: {input_file}")
                return False
            
            with open(input_path, 'r', encoding='utf-8') as f:
                self.challenge_data = json.load(f)
            
            # Set challenge mode
            self.challenge_mode = True
            
            # Extract persona and job from challenge data
            persona_info = self.challenge_data.get('persona', {})
            job_info = self.challenge_data.get('job_to_be_done', {})
            
            # Convert persona to expected format
            self.persona_data = {
                "role": persona_info.get('role', 'User')
            }
            
            # Set job to be done
            self.job_to_be_done = job_info.get('task', '')
            
            # Set PDF base path (assume PDFs folder is in same directory as input file)
            self.pdf_base_path = input_path.parent / "PDFs"
            
            logger.info(f"Loaded challenge input from: {input_file}")
            logger.info(f"Persona role: {self.persona_data['role']}")
            logger.info(f"Job to be done: {self.job_to_be_done}")
            logger.info(f"Documents to process: {len(self.challenge_data.get('documents', []))}")
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in challenge input file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading challenge input file: {e}")
            return False
    
    def get_challenge_pdf_files(self) -> List[Path]:
        """
        Get PDF files specified in challenge input.
        
        Returns:
            List of PDF file paths from challenge input
        """
        if not self.challenge_mode or not self.challenge_data:
            return []
        
        pdf_files = []
        documents = self.challenge_data.get('documents', [])
        
        for doc in documents:
            filename = doc.get('filename', '')
            if filename:
                pdf_path = self.pdf_base_path / filename
                if pdf_path.exists():
                    pdf_files.append(pdf_path)
                    logger.info(f"Found PDF: {filename}")
                else:
                    logger.warning(f"PDF not found: {pdf_path}")
        
        logger.info(f"Found {len(pdf_files)} PDF files for challenge")
        return pdf_files
    
    def load_chunks_from_file(self, chunks_file: Path) -> List[Dict[str, Any]]:
        """
        Load chunks from a JSON file.
        
        Args:
            chunks_file: Path to chunks JSON file
            
        Returns:
            List of chunk dictionaries
        """
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            if not isinstance(chunks, list):
                logger.error(f"Invalid chunks format in {chunks_file}")
                return []
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading chunks from {chunks_file}: {e}")
            return []
    
    def build_master_chunks(self) -> bool:
        """
        Build master list of all chunks from all processed PDFs.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Building master chunk list from all processed PDFs...")
        
        self.master_chunks = []
        chunk_files = list(self.chunks_dir.glob("*_chunks.json"))
        
        if not chunk_files:
            logger.error("No chunk files found to build master list")
            return False
        
        for chunk_file in chunk_files:
            chunks = self.load_chunks_from_file(chunk_file)
            if chunks:
                self.master_chunks.extend(chunks)
                logger.info(f"Added {len(chunks)} chunks from {chunk_file.name}")
        
        logger.info(f"Master chunk list built with {len(self.master_chunks)} total chunks")
        return len(self.master_chunks) > 0
    
    def initialize_retrieval_components(self) -> bool:
        """
        Initialize retriever and reranker components.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing retrieval components...")
            
            # Initialize retriever
            self.retriever = Retriever(model_name=self.retriever_model)
            
            # Initialize reranker
            self.reranker = Reranker(model_name=self.reranker_model)
            
            logger.info("Retrieval components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize retrieval components: {e}")
            return False
    
    def build_search_index(self) -> bool:
        """
        Build search index from master chunks.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.master_chunks:
            logger.error("No chunks available to build index")
            return False
        
        try:
            logger.info(f"Building search index with {len(self.master_chunks)} chunks...")
            self.retriever.build_index(self.master_chunks)
            logger.info("Search index built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build search index: {e}")
            return False
    
    def formulate_persona_query(self) -> str:
        """
        Formulate the full persona-based query string.
        
        Returns:
            Complete query string combining persona and job-to-be-done
        """
        query_parts = []
        
        # Add job-to-be-done if provided
        if self.job_to_be_done:
            query_parts.append(self.job_to_be_done)
        
        # Add persona context if available
        if self.persona_data:
            # Extract relevant persona information
            if 'role' in self.persona_data:
                query_parts.append(f"Role: {self.persona_data['role']}")
            
            if 'responsibilities' in self.persona_data:
                responsibilities = self.persona_data['responsibilities']
                if isinstance(responsibilities, list):
                    query_parts.append(f"Responsibilities: {', '.join(responsibilities)}")
                else:
                    query_parts.append(f"Responsibilities: {responsibilities}")
            
            if 'goals' in self.persona_data:
                goals = self.persona_data['goals']
                if isinstance(goals, list):
                    query_parts.append(f"Goals: {', '.join(goals)}")
                else:
                    query_parts.append(f"Goals: {goals}")
            
            if 'context' in self.persona_data:
                query_parts.append(f"Context: {self.persona_data['context']}")
        
        # Combine all parts
        full_query = " ".join(query_parts)
        
        # Fallback if no query could be formulated
        if not full_query.strip():
            full_query = "relevant information"
        
        logger.info(f"Formulated query: {full_query[:100]}...")
        return full_query
    
    def perform_retrieval_and_reranking(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform complete retrieval and reranking pipeline.
        
        Args:
            query: Search query string
            
        Returns:
            List of reranked results with importance rankings
        """
        try:
            # Step 1: Initial retrieval
            logger.info(f"Performing initial retrieval (top {self.initial_candidates})...")
            initial_results = self.retriever.search(query, top_n=self.initial_candidates)
            
            if not initial_results:
                logger.warning("No initial results found")
                return []
            
            logger.info(f"Retrieved {len(initial_results)} initial candidates")
            
            # Step 2: Reranking
            logger.info(f"Reranking to top {self.final_results} results...")
            final_results = self.reranker.rerank(query, initial_results, top_n=self.final_results)
            
            logger.info(f"Reranking complete. Final results: {len(final_results)}")
            
            # Step 3: Add importance rankings
            for i, result in enumerate(final_results, 1):
                result['importance_rank'] = i
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during retrieval and reranking: {e}")
            return []
    
    def format_output(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Format results into the required JSON output format.
        
        Args:
            results: List of reranked results
            query: Original query string
            
        Returns:
            Formatted output dictionary
        """
        formatted_output = {
            "query": query,
            "persona": self.persona_data,
            "job_to_be_done": self.job_to_be_done,
            "total_results": len(results),
            "retrieval_metadata": {
                "retriever_model": self.retriever_model,
                "reranker_model": self.reranker_model,
                "initial_candidates": self.initial_candidates,
                "final_results": self.final_results,
                "total_chunks_indexed": len(self.master_chunks)
            },
            "results": []
        }
        
        for result in results:
            formatted_result = {
                "importance_rank": result['importance_rank'],
                "reranker_score": result['reranker_score'],
                "similarity_score": result.get('similarity_score', 0.0),
                "content": {
                    "text_content": result['text_content'],
                    "heading_text": result['heading_text'],
                    "heading_level": result['heading_level']
                },
                "metadata": {
                    "source_document": result['source_document'],
                    "page_number": result['page_number']
                }
            }
            formatted_output["results"].append(formatted_result)
        
        return formatted_output
    
    def format_challenge_output(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Format results into challenge output format.
        
        Args:
            results: List of reranked results
            query: Original query string
            
        Returns:
            Formatted output dictionary in challenge format
        """
        from datetime import datetime
        
        # Get document filenames from challenge data
        input_documents = []
        if self.challenge_data:
            documents = self.challenge_data.get('documents', [])
            input_documents = [doc.get('filename', '') for doc in documents]
        
        # Create metadata
        metadata = {
            "input_documents": input_documents,
            "persona": self.persona_data.get('role', 'User') if self.persona_data else 'User',
            "job_to_be_done": self.job_to_be_done or '',
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Create extracted sections (top-level results)
        extracted_sections = []
        for result in results:
            section = {
                "document": Path(result['source_document']).name,
                "section_title": result['heading_text'],
                "importance_rank": result['importance_rank'],
                "page_number": result['page_number']
            }
            extracted_sections.append(section)
        
        # Create subsection analysis (detailed content)
        subsection_analysis = []
        for result in results:
            subsection = {
                "document": Path(result['source_document']).name,
                "refined_text": result['text_content'],
                "page_number": result['page_number']
            }
            subsection_analysis.append(subsection)
        
        return {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
    
    def process_all_pdfs_legacy(self) -> Dict[str, Any]:
        """
        Legacy method: Process all PDFs without search functionality.
        
        Returns:
            Dictionary with processing results and statistics
        """
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return {
                'total_files': 0,
                'processed': 0,
                'failed': 0,
                'processing_time': 0
            }
        
        start_time = time.time()
        logger.info(f"Starting batch processing of {len(pdf_files)} PDF files")
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Progress: {i}/{len(pdf_files)}")
            self.process_single_pdf(pdf_path)
        
        total_time = time.time() - start_time
        
        # Generate summary
        results = {
            'total_files': len(pdf_files),
            'processed': len(self.processed_files),
            'failed': len(self.failed_files),
            'processing_time': total_time,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'processing_stats': self.processing_stats
        }
        
        # Save processing report
        report_file = self.output_dir / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing complete! Report saved to: {report_file}")
        
        return results
    
    def process_single_pdf(self, pdf_path: Path) -> bool:
        """
        Process a single PDF through the complete pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        logger.info(f"Processing: {pdf_path.name}")
        
        try:
            # Step 1: Extract headings
            headings_file = self.extract_headings(pdf_path)
            if not headings_file:
                self.failed_files.append(str(pdf_path))
                return False
            
            # Step 2: Create chunks
            chunks_file = self.create_chunks(headings_file, pdf_path)
            if not chunks_file:
                self.failed_files.append(str(pdf_path))
                # Clean up temporary headings file
                if not self.keep_intermediates and headings_file.exists():
                    headings_file.unlink()
                return False
            
            # Clean up temporary headings file if not keeping intermediates
            if not self.keep_intermediates and headings_file.exists() and headings_file.name.startswith("temp_"):
                headings_file.unlink()
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully processed {pdf_path.name} in {processing_time:.2f} seconds")
            
            self.processed_files.append(str(pdf_path))
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            self.failed_files.append(str(pdf_path))
            return False
    
    def process_all_pdfs_and_search(self) -> Dict[str, Any]:
        """
        Complete pipeline: process PDFs, build index, and perform persona-based search.
        
        Returns:
            Dictionary with processing results and search results
        """
        start_time = time.time()
        
        # Step 1: Load persona data (skip if challenge mode, already loaded)
        if not self.challenge_mode and not self.load_persona():
            return {"error": "Failed to load persona data"}
        
        # Step 2: Process PDFs
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return {
                'error': 'No PDF files found',
                'total_files': 0,
                'processed': 0,
                'failed': 0,
                'processing_time': 0
            }
        
        logger.info(f"Starting batch processing of {len(pdf_files)} PDF files")
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Progress: {i}/{len(pdf_files)}")
            self.process_single_pdf(pdf_path)
        
        processing_time = time.time() - start_time
        
        # Step 3: Build master chunk list
        if not self.build_master_chunks():
            return {"error": "Failed to build master chunk list"}
        
        # Step 4: Initialize retrieval components
        if not self.initialize_retrieval_components():
            return {"error": "Failed to initialize retrieval components"}
        
        # Step 5: Build search index
        if not self.build_search_index():
            return {"error": "Failed to build search index"}
        
        # Step 6: Formulate query and perform search
        query = self.formulate_persona_query()
        initial_search_results = self.perform_retrieval_and_reranking(query)
        
        # NEW STEP: Enforce that at least one result from every PDF is included
        search_results = self._enforce_document_diversity(initial_search_results, query)
        
        total_time = time.time() - start_time
        
        # Step 7: Format output
        if self.challenge_mode:
            formatted_output = self.format_challenge_output(search_results, query)
        else:
            formatted_output = self.format_output(search_results, query)
        
        # Add processing statistics (only for non-challenge mode)
        if not self.challenge_mode:
            formatted_output["processing_stats"] = {
                'total_files': len(pdf_files),
                'processed': len(self.processed_files),
                'failed': len(self.failed_files),
                'processing_time': processing_time,
                'total_time': total_time,
                'processed_files': self.processed_files,
                'failed_files': self.failed_files,
                'chunk_stats': self.processing_stats
            }
        
        # Save processing report (only for non-challenge mode)
        if not self.challenge_mode:
            report_file = self.output_dir / "processing_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_output["processing_stats"], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Complete pipeline finished in {total_time:.2f} seconds")
        
        return formatted_output
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of processing results.
        
        Args:
            results: Results dictionary from process_all_pdfs
        """
        print("\n" + "="*60)
        print("PDF PROCESSING SUMMARY")
        print("="*60)
        
        print(f"Total files found: {results['total_files']}")
        print(f"Successfully processed: {results['processed']}")
        print(f"Failed: {results['failed']}")
        print(f"Total processing time: {results['processing_time']:.2f} seconds")
        
        if results['processed'] > 0:
            avg_time = results['processing_time'] / results['processed']
            print(f"Average time per file: {avg_time:.2f} seconds")
        
        if results['failed_files']:
            print(f"\nFailed files:")
            for file in results['failed_files']:
                print(f"  - {file}")
        
        if results['processing_stats']:
            print(f"\nChunk Statistics:")
            total_chunks = sum(stats['num_chunks'] for stats in results['processing_stats'].values())
            total_text = sum(stats['total_text_length'] for stats in results['processing_stats'].values())
            
            print(f"  Total chunks created: {total_chunks}")
            print(f"  Total text processed: {total_text:,} characters")
            
            if total_chunks > 0:
                avg_chunk_size = total_text / total_chunks
                print(f"  Average chunk size: {avg_chunk_size:.0f} characters")
        
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Chunks saved in: {self.chunks_dir}")
        
        if self.keep_intermediates:
            print(f"Headings saved in: {self.headings_dir}")


def main():
    """
    Main function to handle command line arguments and run the enhanced processor.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced PDF processing with persona-based retrieval and reranking"
    )
    
    parser.add_argument(
        "input_source",
        help="Path to folder containing PDF files OR path to challenge input JSON file"
    )
    
    parser.add_argument(
        "--persona", "-p",
        help="Path to persona JSON file (ignored if using challenge input)"
    )
    
    parser.add_argument(
        "--job-to-be-done", "-j",
        help="Job description or task to be accomplished (ignored if using challenge input)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="chunks",
        help="Output directory for processed files (default: chunks)"
    )
    
    parser.add_argument(
        "--output-file", "-f",
        help="Output JSON file for search results (default: results.json)"
    )
    
    parser.add_argument(
        "--keep-intermediates", "-k",
        action="store_true",
        help="Keep intermediate heading JSON files"
    )
    
    parser.add_argument(
        "--retriever-model", "-r",
        default="BAAI/bge-small-en-v1.5",
        help="Bi-encoder model for initial retrieval"
    )
    
    parser.add_argument(
        "--reranker-model", "-m",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking"
    )
    
    parser.add_argument(
        "--initial-candidates", "-i",
        type=int,
        default=50,
        help="Number of initial candidates to retrieve (default: 50)"
    )
    
    parser.add_argument(
        "--final-results", "-n",
        type=int,
        default=20,
        help="Number of final results to return (default: 20)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process PDFs without performing search (legacy mode)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Detect if input is a challenge JSON file or PDF folder
    input_path = Path(args.input_source)
    is_challenge_input = input_path.is_file() and input_path.suffix.lower() == '.json'
    
    if is_challenge_input:
        # Challenge mode - validate JSON file exists
        if not input_path.exists():
            print(f"Error: Challenge input file does not exist: {args.input_source}")
            sys.exit(1)
        pdf_folder = None
    else:
        # Regular mode - validate PDF folder
        if not input_path.exists():
            print(f"Error: PDF folder does not exist: {args.input_source}")
            sys.exit(1)
        pdf_folder = args.input_source
        
        # Validate persona and job requirements for search mode
        if not args.process_only and not args.persona and not args.job_to_be_done:
            print("Error: Either --persona or --job-to-be-done (or both) must be provided for search mode")
            print("Use --process-only to run in legacy mode without search")
            sys.exit(1)
    
    # Initialize enhanced processor
    processor = EnhancedPDFProcessor(
        pdf_folder=pdf_folder or ".",  # Dummy folder for challenge mode
        persona_file=args.persona,
        job_to_be_done=args.job_to_be_done,
        output_dir=args.output_dir,
        keep_intermediates=args.keep_intermediates,
        retriever_model=args.retriever_model,
        reranker_model=args.reranker_model,
        initial_candidates=args.initial_candidates,
        final_results=args.final_results
    )
    
    # Load challenge input if in challenge mode
    if is_challenge_input:
        if not processor.load_challenge_input(args.input_source):
            print("Error: Failed to load challenge input file")
            sys.exit(1)
    
    try:
        if args.process_only and not is_challenge_input:
            # Legacy mode: just process PDFs
            logger.info("Running in process-only mode (legacy)")
            results = processor.process_all_pdfs_legacy()
            processor.print_summary(results)
            
            if results['failed'] > 0:
                sys.exit(1)
        else:
            # Enhanced mode: process PDFs and perform search
            if is_challenge_input:
                logger.info("Running in challenge mode with retrieval and reranking")
            else:
                logger.info("Running enhanced mode with retrieval and reranking")
                
            results = processor.process_all_pdfs_and_search()
            
            if "error" in results:
                print(f"Error: {results['error']}")
                sys.exit(1)
            
            # Save results to file
            output_file = args.output_file or "results.json"
            output_path = Path(output_file)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            if is_challenge_input:
                print("CHALLENGE MODE PROCESSING COMPLETE")
            else:
                print("ENHANCED PDF PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Results saved to: {output_path}")
            
            if is_challenge_input:
                # Challenge mode output
                print(f"Extracted sections: {len(results.get('extracted_sections', []))}")
                print(f"Subsection analysis: {len(results.get('subsection_analysis', []))}")
                print(f"Persona: {results.get('metadata', {}).get('persona', 'N/A')}")
                print(f"Job to be done: {results.get('metadata', {}).get('job_to_be_done', 'N/A')}")
                
                if results.get('extracted_sections'):
                    print(f"\nTop 3 Extracted Sections:")
                    for section in results['extracted_sections'][:3]:
                        print(f"{section['importance_rank']}. {section['document']} - {section['section_title'][:60]}...")
            else:
                # Regular enhanced mode output
                print(f"Total results: {results['total_results']}")
                print(f"Query: {results['query'][:100]}...")
                
                if results['total_results'] > 0:
                    print(f"\nTop 3 Results:")
                    for i, result in enumerate(results['results'][:3], 1):
                        print(f"{i}. Score: {result['reranker_score']:.4f} - {result['content']['heading_text'][:60]}...")
                
                # Print processing stats (only for non-challenge mode)
                if 'processing_stats' in results:
                    stats = results['processing_stats']
                    print(f"\nProcessing Statistics:")
                    print(f"  Files processed: {stats['processed']}/{stats['total_files']}")
                    print(f"  Total processing time: {stats['total_time']:.2f} seconds")
                    print(f"  Chunks indexed: {results['retrieval_metadata']['total_chunks_indexed']}")
                    
                    if stats['failed'] > 0:
                        print(f"  Failed files: {stats['failed']}")
                        sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()