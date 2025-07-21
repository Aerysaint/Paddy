"""
Logging configuration and error handling framework for PDF Outline Extractor.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("pdf_outline_extractor")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""
    pass


class PDFExtractionError(PDFProcessingError):
    """Exception raised when PDF text extraction fails."""
    pass


class HeadingDetectionError(PDFProcessingError):
    """Exception raised when heading detection fails."""
    pass


class JSONOutputError(PDFProcessingError):
    """Exception raised when JSON output generation fails."""
    pass


def handle_pdf_error(pdf_path: str, error: Exception, logger: logging.Logger) -> None:
    """
    Handle PDF processing errors with appropriate logging.
    
    Args:
        pdf_path: Path to the PDF file that caused the error
        error: The exception that occurred
        logger: Logger instance for error reporting
    """
    error_msg = f"Error processing PDF '{pdf_path}': {str(error)}"
    
    if isinstance(error, PDFProcessingError):
        logger.error(error_msg)
    else:
        logger.exception(error_msg)


def safe_execute(func, *args, default=None, logger: Optional[logging.Logger] = None, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default: Default value to return on error
        logger: Logger instance for error reporting
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default