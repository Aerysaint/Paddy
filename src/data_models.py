"""
Core data models for PDF Outline Extractor.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Any, Optional


@dataclass
class TextBlock:
    """
    Represents a block of text extracted from a PDF with formatting metadata.
    """
    text: str
    page: int
    font_size: float
    font_name: str
    is_bold: bool
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    line_height: float
    
    def __post_init__(self):
        """Validate and normalize text block data."""
        self.text = self.text.strip() if self.text else ""
        self.page = max(0, int(self.page))  # Allow 0-based page numbering
        self.font_size = max(0.0, float(self.font_size))
        self.line_height = max(0.0, float(self.line_height))
    
    @property
    def width(self) -> float:
        """Calculate the width of the text block."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Calculate the height of the text block."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Calculate the area of the text block."""
        return self.width * self.height
    
    def is_empty(self) -> bool:
        """Check if the text block is empty or contains only whitespace."""
        return not self.text or self.text.isspace()


@dataclass
class HeadingCandidate:
    """
    Represents a potential heading with confidence scoring and analysis metadata.
    """
    text: str
    page: int
    confidence_score: float
    formatting_features: Dict[str, Any]
    level_indicators: List[str]
    text_block: Optional[TextBlock] = None
    assigned_level: Optional[int] = None
    
    def __post_init__(self):
        """Validate and normalize heading candidate data."""
        self.text = self.text.strip() if self.text else ""
        self.page = max(0, int(self.page))  # Allow 0-based page numbering
        self.confidence_score = max(0.0, min(1.0, float(self.confidence_score)))
        
        if not isinstance(self.formatting_features, dict):
            self.formatting_features = {}
        
        if not isinstance(self.level_indicators, list):
            self.level_indicators = []
    
    def add_level_indicator(self, indicator: str) -> None:
        """Add a level indicator to the candidate."""
        if indicator and indicator not in self.level_indicators:
            self.level_indicators.append(indicator)
    
    def update_confidence(self, score: float) -> None:
        """Update the confidence score, ensuring it stays within valid range."""
        self.confidence_score = max(0.0, min(1.0, float(score)))
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if the candidate has high confidence above the threshold."""
        return self.confidence_score >= threshold


@dataclass
class DocumentStructure:
    """
    Represents the complete structure of a processed PDF document.
    """
    title: str
    headings: List[Dict[str, Union[str, int]]]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate and normalize document structure data."""
        self.title = self.title.strip() if self.title else ""
        
        if not isinstance(self.headings, list):
            self.headings = []
        
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # Validate heading format
        validated_headings = []
        for heading in self.headings:
            if isinstance(heading, dict) and all(key in heading for key in ['level', 'text', 'page']):
                validated_headings.append({
                    'level': int(heading['level']),
                    'text': str(heading['text']).strip(),
                    'page': max(1, int(heading['page']))
                })
        self.headings = validated_headings
    
    def add_heading(self, level: int, text: str, page: int) -> None:
        """Add a heading to the document structure."""
        if text and text.strip():
            self.headings.append({
                'level': max(1, min(3, int(level))),  # Ensure level is 1-3
                'text': str(text).strip(),
                'page': max(1, int(page))
            })
    
    def get_headings_by_level(self, level: int) -> List[Dict[str, Union[str, int]]]:
        """Get all headings of a specific level."""
        return [h for h in self.headings if h['level'] == level]
    
    def get_headings_by_page(self, page: int) -> List[Dict[str, Union[str, int]]]:
        """Get all headings on a specific page."""
        return [h for h in self.headings if h['page'] == page]
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary matching the output schema."""
        return {
            'title': self.title,
            'outline': [
                {
                    'level': heading['level'],
                    'text': heading['text'],
                    'page': heading['page']
                }
                for heading in self.headings
            ]
        }
    
    def is_empty(self) -> bool:
        """Check if the document structure is empty."""
        return not self.title and not self.headings


@dataclass
class ProcessingStats:
    """
    Statistics and metrics for PDF processing operations.
    """
    total_text_blocks: int = 0
    heading_candidates: int = 0
    confirmed_headings: int = 0
    processing_time: float = 0.0
    pages_processed: int = 0
    errors_encountered: int = 0
    
    def __post_init__(self):
        """Validate statistics data."""
        self.total_text_blocks = max(0, int(self.total_text_blocks))
        self.heading_candidates = max(0, int(self.heading_candidates))
        self.confirmed_headings = max(0, int(self.confirmed_headings))
        self.processing_time = max(0.0, float(self.processing_time))
        self.pages_processed = max(0, int(self.pages_processed))
        self.errors_encountered = max(0, int(self.errors_encountered))
    
    @property
    def heading_detection_rate(self) -> float:
        """Calculate the rate of confirmed headings from candidates."""
        if self.heading_candidates == 0:
            return 0.0
        return self.confirmed_headings / self.heading_candidates
    
    @property
    def processing_speed(self) -> float:
        """Calculate pages processed per second."""
        if self.processing_time == 0.0:
            return 0.0
        return self.pages_processed / self.processing_time