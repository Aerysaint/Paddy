"""
Main entry point for PDF Outline Extractor.

This module serves as the primary application entry point, integrating all components
into a complete processing pipeline with command-line argument handling and
performance optimizations.
"""

import sys
import argparse
import os
import time
import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.logging_config import setup_logging, PDFProcessingError, handle_pdf_error
from src.batch_processor import BatchProcessor


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the PDF Outline Extractor.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="PDF Outline Extractor - Extract document structure from PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default directories (/app/input, /app/output)
  %(prog)s -i ./pdfs -o ./output             # Custom input/output directories
  %(prog)s -i ./pdfs -o ./output -v          # Verbose logging
  %(prog)s -i ./pdfs -o ./output --workers 2 # Use 2 worker threads
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='/app/input',
        help='Input directory containing PDF files (default: /app/input)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/app/output',
        help='Output directory for JSON files (default: /app/output)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker threads for parallel processing (default: 1)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--single-file',
        type=str,
        help='Process a single PDF file instead of directory batch processing'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file path for single file processing (used with --single-file)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Single file processing validation
    if args.single_file:
        if not Path(args.single_file).exists():
            print(f"Error: Input file does not exist: {args.single_file}", file=sys.stderr)
            return False
        
        if not Path(args.single_file).suffix.lower() == '.pdf':
            print(f"Error: Input file is not a PDF: {args.single_file}", file=sys.stderr)
            return False
        
        if args.output_file and Path(args.output_file).is_dir():
            print(f"Error: Output file path is a directory: {args.output_file}", file=sys.stderr)
            return False
    
    # Directory processing validation
    else:
        if not Path(args.input).exists():
            print(f"Error: Input directory does not exist: {args.input}", file=sys.stderr)
            return False
        
        if not Path(args.input).is_dir():
            print(f"Error: Input path is not a directory: {args.input}", file=sys.stderr)
            return False
    
    # Worker count validation
    if args.workers < 1 or args.workers > 8:
        print(f"Error: Worker count must be between 1 and 8, got: {args.workers}", file=sys.stderr)
        return False
    
    return True


def setup_environment(args: argparse.Namespace) -> logging.Logger:
    """
    Set up the application environment and logging.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Configured logger instance
    """
    # Determine log level
    log_level = 'DEBUG' if args.verbose else args.log_level
    
    # Set up logging
    logger = setup_logging(log_level)
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("PDF OUTLINE EXTRACTOR STARTING")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Workers: {args.workers}")
    
    if args.single_file:
        logger.info(f"Single file mode: {args.single_file}")
        if args.output_file:
            logger.info(f"Output file: {args.output_file}")
    
    return logger


def create_output_directory(output_path: str, logger: logging.Logger) -> bool:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path: Path to output directory
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create output directory {output_path}: {str(e)}")
        return False


def process_single_file(input_file: str, output_file: Optional[str], logger: logging.Logger) -> int:
    """
    Process a single PDF file.
    
    Args:
        input_file: Path to input PDF file
        output_file: Optional path to output JSON file
        logger: Logger instance
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Determine output file path
        if output_file:
            output_path = output_file
        else:
            input_path = Path(input_file)
            output_path = str(input_path.with_suffix('.json'))
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if not create_output_directory(str(output_dir), logger):
            return 1
        
        logger.info(f"Processing single file: {input_file} -> {output_path}")
        
        # Create batch processor and process single file
        processor = BatchProcessor(max_workers=1)
        success = processor.process_single_pdf(input_file, output_path)
        
        if success:
            logger.info("Single file processing completed successfully")
            return 0
        else:
            logger.error("Single file processing failed")
            return 1
            
    except Exception as e:
        handle_pdf_error(input_file, e, logger)
        return 1


def process_directory_batch(input_dir: str, output_dir: str, workers: int, logger: logging.Logger) -> int:
    """
    Process all PDF files in a directory.
    
    Args:
        input_dir: Path to input directory
        output_dir: Path to output directory
        workers: Number of worker threads
        logger: Logger instance
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Create output directory
        if not create_output_directory(output_dir, logger):
            return 1
        
        logger.info(f"Starting batch processing: {input_dir} -> {output_dir}")
        
        # Create batch processor
        processor = BatchProcessor(max_workers=workers)
        
        # Process directory
        start_time = time.time()
        stats = processor.process_directory(input_dir, output_dir)
        processing_time = time.time() - start_time
        
        # Log final results
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total files: {stats.get('total_files', 0)}")
        logger.info(f"Successful: {stats.get('successful', 0)}")
        logger.info(f"Failed: {stats.get('failed', 0)}")
        logger.info(f"Total time: {processing_time:.2f} seconds")
        
        if stats.get('total_files', 0) > 0:
            success_rate = (stats.get('successful', 0) / stats.get('total_files', 1)) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Determine exit code based on results
        if stats.get('failed', 0) == 0:
            logger.info("All files processed successfully")
            return 0
        elif stats.get('successful', 0) > 0:
            logger.warning("Some files failed but processing completed")
            return 0  # Partial success is still considered success for batch processing
        else:
            logger.error("All files failed to process")
            return 1
            
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        logger.debug(f"Full error details:", exc_info=True)
        return 1


def optimize_memory_usage() -> None:
    """
    Apply basic memory optimizations for container environment.
    """
    # Force garbage collection
    gc.collect()
    
    # Set garbage collection thresholds for better memory management
    # More aggressive collection for container environment
    gc.set_threshold(700, 10, 10)


def check_system_resources(logger: logging.Logger) -> None:
    """
    Check and log system resource information.
    
    Args:
        logger: Logger instance
    """
    try:
        import psutil
        
        # Memory information
        memory = psutil.virtual_memory()
        logger.debug(f"Available memory: {memory.available / (1024**3):.1f} GB")
        logger.debug(f"Memory usage: {memory.percent}%")
        
        # CPU information
        cpu_count = psutil.cpu_count()
        logger.debug(f"CPU cores: {cpu_count}")
        
    except ImportError:
        # psutil not available, skip resource checking
        logger.debug("psutil not available, skipping resource check")
    except Exception as e:
        logger.debug(f"Error checking system resources: {str(e)}")


def main() -> int:
    """
    Main application entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Validate arguments
        if not validate_arguments(args):
            return 1
        
        # Set up environment and logging
        logger = setup_environment(args)
        
        # Apply memory optimizations
        optimize_memory_usage()
        
        # Check system resources (debug info)
        check_system_resources(logger)
        
        # Process files based on mode
        if args.single_file:
            # Single file processing mode
            exit_code = process_single_file(args.single_file, args.output_file, logger)
        else:
            # Directory batch processing mode
            exit_code = process_directory_batch(args.input, args.output, args.workers, logger)
        
        # Final cleanup
        optimize_memory_usage()
        
        # Log completion
        if exit_code == 0:
            logger.info("PDF Outline Extractor completed successfully")
        else:
            logger.error("PDF Outline Extractor completed with errors")
        
        logger.info("=" * 60)
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        # Fallback error handling for unexpected errors
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())