#!/usr/bin/env python3
"""
PDF Layout Extractor - Main orchestration module

This module provides the main entry point for processing PDF files and extracting
structured outlines containing titles and hierarchical headings.
"""

import argparse
import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import pipeline modules
from parser import extract_blocks
from cluster import cluster_font_sizes, assign_size_levels
from detect import filter_headers_footers, detect_title, detect_headings
from outline import build_hierarchy, format_json_structure, serialize_to_json

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for input and output directories.
    
    Returns:
        argparse.Namespace: Parsed arguments with input and output paths
    """
    parser = argparse.ArgumentParser(
        description='Extract structured outlines from PDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input /path/to/pdfs --output /path/to/output
  python main.py -i ./pdfs -o ./results
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing PDF files to process'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str,
        required=True,
        help='Output directory for generated JSON files'
    )
    
    return parser.parse_args()


def validate_input_directory(input_path: str) -> Path:
    """
    Validate that input directory exists and contains PDF files.
    
    Args:
        input_path: Path to input directory
        
    Returns:
        Path: Validated input directory path
        
    Raises:
        SystemExit: If directory doesn't exist or contains no PDF files
    """
    input_dir = Path(input_path)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_path}")
        sys.exit(1)
    
    # Check for PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in input directory: {input_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_files)} PDF files in input directory: {input_path}")
    return input_dir


def create_output_directory(output_path: str) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path: Path to output directory
        
    Returns:
        Path: Output directory path
        
    Raises:
        SystemExit: If directory cannot be created
    """
    output_dir = Path(output_path)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_path}")
        return output_dir
    except OSError as e:
        logger.error(f"Failed to create output directory {output_path}: {e}")
        sys.exit(1)


def discover_pdf_files(input_dir: Path) -> List[Path]:
    """
    Scan input directory for all *.pdf files.
    
    Args:
        input_dir: Input directory path
        
    Returns:
        List of PDF file paths found in the directory
    """
    pdf_files = list(input_dir.glob("*.pdf"))
    pdf_files.sort()  # Sort for consistent processing order
    
    logger.info(f"Discovered {len(pdf_files)} PDF files for processing")
    for pdf_file in pdf_files:
        logger.debug(f"Found PDF: {pdf_file.name}")
    
    return pdf_files


def process_single_pdf(pdf_path: Path, output_dir: Path) -> bool:
    """
    Process a single PDF through the complete pipeline with comprehensive error handling.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for JSON file
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Validate PDF file exists and is readable
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path.name}")
            return False
        
        if not pdf_path.is_file():
            logger.error(f"Path is not a file: {pdf_path.name}")
            return False
        
        # Step 1: Extract text blocks from PDF
        logger.debug(f"Extracting blocks from {pdf_path.name}")
        try:
            blocks = extract_blocks(str(pdf_path))
            logger.debug(f"Extracted {len(blocks)} blocks from {pdf_path.name}")
        except Exception as e:
            # Check if this is a corrupted PDF error
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['no /root object', 'not a pdf', 'corrupted', 'invalid pdf']):
                logger.warning(f"Corrupted or invalid PDF detected: {pdf_path.name} - {e}")
                logger.warning(f"Creating empty output for {pdf_path.name}")
                # Create empty output for corrupted PDFs
                output_data = {"title": "", "outline": []}
                # Generate output filename and write empty JSON
                output_filename = pdf_path.stem + ".json"
                output_path = output_dir / output_filename
                try:
                    json_content = serialize_to_json(output_data)
                    output_path.write_text(json_content, encoding='utf-8')
                    logger.info(f"Created empty output for corrupted PDF: {pdf_path.name} -> {output_filename}")
                    return True
                except Exception as write_error:
                    logger.error(f"Failed to write empty output for {pdf_path.name}: {write_error}")
                    return False
            else:
                logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
                return False
        
        if not blocks:
            logger.warning(f"No text blocks found in {pdf_path.name} - creating empty output")
            # Create empty output for PDFs with no extractable text
            output_data = {"title": "", "outline": []}
        else:
            try:
                # Step 2: Filter headers/footers
                logger.debug(f"Filtering headers/footers for {pdf_path.name}")
                filtered_blocks, blocks_with_flags = filter_headers_footers(blocks)
                logger.debug(f"Filtered to {len(filtered_blocks)} blocks after header/footer removal")
                
                # Step 3: Cluster font sizes and assign levels
                logger.debug(f"Clustering font sizes for {pdf_path.name}")
                size_to_level = cluster_font_sizes(filtered_blocks)
                blocks_with_levels = assign_size_levels(filtered_blocks, size_to_level)
                logger.debug(f"Assigned size levels to {len(blocks_with_levels)} blocks")
                
                # Step 4: Detect title
                logger.debug(f"Detecting title for {pdf_path.name}")
                title = detect_title(blocks_with_levels)
                logger.debug(f"Detected title: '{title}'" if title else "No title detected")
                
                # Step 5: Detect headings
                logger.debug(f"Detecting headings for {pdf_path.name}")
                headings = detect_headings(blocks_with_levels)
                logger.debug(f"Detected {len(headings)} headings")
                
                # Step 6: Build hierarchy
                logger.debug(f"Building hierarchy for {pdf_path.name}")
                hierarchy = build_hierarchy(headings)
                logger.debug(f"Built hierarchy with {len(hierarchy)} structured headings")
                
                # Step 7: Format JSON output (pass original headings, not hierarchy)
                output_data = format_json_structure(title, headings)
                
            except Exception as e:
                logger.error(f"Failed during processing pipeline for {pdf_path.name}: {e}")
                # Create empty output as fallback
                logger.warning(f"Creating empty output for {pdf_path.name} due to processing error")
                output_data = {"title": "", "outline": []}
        
        # Generate output filename (filename.json for filename.pdf)
        output_filename = pdf_path.stem + ".json"
        output_path = output_dir / output_filename
        
        # Write JSON output with error handling
        try:
            json_content = serialize_to_json(output_data)
            output_path.write_text(json_content, encoding='utf-8')
            logger.info(f"Successfully processed {pdf_path.name} -> {output_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to write output file for {pdf_path.name}: {e}")
            return False
        
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path.name}")
        return False
    except PermissionError:
        logger.error(f"Permission denied accessing {pdf_path.name}")
        return False
    except OSError as e:
        logger.error(f"OS error processing {pdf_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {pdf_path.name}: {e}")
        logger.debug(f"Error details for {pdf_path.name}: {type(e).__name__}: {str(e)}")
        return False


def process_pdf_files(pdf_files: List[Path], output_dir: Path) -> None:
    """
    Process each PDF through the complete pipeline and generate JSON outputs.
    
    Args:
        pdf_files: List of PDF file paths to process
        output_dir: Output directory for JSON files
    """
    if not pdf_files:
        logger.warning("No PDF files to process")
        return
    
    successful_count = 0
    failed_count = 0
    failed_files = []
    
    total_files = len(pdf_files)
    logger.info(f"Starting processing of {total_files} PDF files")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"Processing file {i}/{total_files}: {pdf_path.name}")
        
        success = process_single_pdf(pdf_path, output_dir)
        if success:
            successful_count += 1
            logger.info(f"✓ Completed {pdf_path.name} ({i}/{total_files})")
        else:
            failed_count += 1
            failed_files.append(pdf_path.name)
            logger.error(f"✗ Failed {pdf_path.name} ({i}/{total_files})")
    
    # Log final results with detailed summary
    logger.info("=" * 60)
    logger.info(f"PROCESSING SUMMARY:")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {failed_count}")
    
    if failed_files:
        logger.warning(f"Failed files: {', '.join(failed_files)}")
    
    # Determine exit behavior based on results
    if successful_count == 0 and failed_count > 0:
        logger.error("All PDF files failed to process - exiting with error code")
        sys.exit(1)
    elif failed_count > 0:
        logger.warning(f"{failed_count} out of {total_files} PDF files failed to process")
        # Continue with success exit code since some files were processed successfully
    else:
        logger.info("All PDF files processed successfully")


def main():
    """Main entry point for PDF layout extraction."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Validate input directory and check for PDF files
        input_dir = validate_input_directory(args.input)
        
        # Create output directory if needed
        output_dir = create_output_directory(args.output)
        
        logger.info("Command-line arguments parsed and validated successfully")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Discover and process PDF files
        pdf_files = discover_pdf_files(input_dir)
        process_pdf_files(pdf_files, output_dir)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()