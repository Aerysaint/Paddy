"""
PDF text extraction and block formation module.

This module handles extracting words from PDF pages using pdfplumber,
merging adjacent words into text blocks, and computing block features.
"""

import pdfplumber
import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)



def extract_words_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract words from PDF pages using pdfplumber with styling attributes.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of word dictionaries with fontname, size, adv attributes and bounding boxes
    """
    words = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Use 1-based page numbering consistently
                page_number = page_num + 1
                # Extract words with detailed attributes
                # Try different extraction parameters to get proper words instead of characters
                page_words = page.extract_words(
                    x_tolerance=3,  # Increased tolerance to group characters into words
                    y_tolerance=3,  # Increased tolerance for vertical alignment
                    keep_blank_chars=False,
                    use_text_flow=False,
                    horizontal_ltr=True,
                    vertical_ttb=True,
                    extra_attrs=['fontname', 'size', 'adv']
                )
                
                # Process each word and add page information
                for word in page_words:
                    # Ensure consistent coordinate mapping: word['top'] → y0, word['bottom'] → y1
                    y0 = word['top']
                    y1 = word['bottom']
                    
                    # Validate coordinate consistency: y0 should be less than y1 (top < bottom)
                    if y0 > y1:
                        logger.warning(f"Coordinate inconsistency detected on page {page_number}: "
                                     f"y0={y0} > y1={y1} for word '{word['text']}'. Swapping coordinates.")
                        y0, y1 = y1, y0  # Swap to maintain consistency
                    
                    word_dict = {
                        'text': word['text'],
                        'page': page_number,
                        'x0': word['x0'],
                        'y0': y0,  # Consistently use 'top' as y0
                        'x1': word['x1'],
                        'y1': y1,  # Consistently use 'bottom' as y1
                        'fontname': word.get('fontname', ''),
                        'size': word.get('size', 0.0),
                        'adv': word.get('adv', 0.0)
                    }
                    words.append(word_dict)
                    
    except Exception as e:
        print(f"Error extracting words from {pdf_path}: {e}")
        return []
    
    return words


def validate_coordinate_consistency(blocks: List[Dict], stage: str = "unknown") -> None:
    """
    Validate that all blocks have consistent coordinate system: y0 < y1.
    
    Args:
        blocks: List of text blocks to validate
        stage: Processing stage name for logging context
    """
    inconsistent_count = 0
    
    for i, block in enumerate(blocks):
        y0 = block.get('y0', 0)
        y1 = block.get('y1', 0)
        
        if y0 > y1:
            inconsistent_count += 1
            logger.error(f"Coordinate inconsistency at {stage} stage, block {i}: "
                       f"y0={y0} > y1={y1} for text '{block.get('text', '')[:30]}...'")
    
    if inconsistent_count > 0:
        logger.error(f"Found {inconsistent_count} coordinate inconsistencies at {stage} stage")
    else:
        logger.debug(f"Coordinate validation passed at {stage} stage: {len(blocks)} blocks checked")


def extract_blocks(pdf_path: str) -> List[Dict]:
    """
    Extract styled text blocks from PDF pages.
    
    This is the main entry point that extracts words and processes them
    into blocks with full feature computation.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of text block dictionaries with all computed features
    """
    # Extract words first
    words = extract_words_from_pdf(pdf_path)
    if not words:
        return []
    
    # Group words by page for processing
    pages = {}
    for word in words:
        page_num = word['page']
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(word)
    
    all_blocks = []
    
    # Process each page
    for page_num in sorted(pages.keys()):
        page_words = pages[page_num]
        
        # Sort words by y-coordinate (top to bottom), then x-coordinate (left to right)
        page_words.sort(key=lambda w: (w['y0'], w['x0']))
        
        # Merge adjacent words into blocks
        page_blocks = merge_adjacent_words(page_words)
        
        # Validate coordinate consistency after merging
        validate_coordinate_consistency(page_blocks, f"merge_page_{page_num}")
        
        # Calculate page average line height
        page_avg_line_height = compute_page_line_height(page_blocks)
        
        # Compute features for each block
        for i, block in enumerate(page_blocks):
            prev_block = page_blocks[i-1] if i > 0 else None
            enhanced_block = compute_block_features(block, prev_block, page_avg_line_height)
            all_blocks.append(enhanced_block)
    
    # Final coordinate validation for all blocks
    validate_coordinate_consistency(all_blocks, "final_extraction")
    
    return all_blocks


def clean_duplicated_characters(text: str) -> str:
    """
    Clean up duplicated characters that may result from PDF rendering issues.
    
    Args:
        text: Input text that may contain duplicated characters
        
    Returns:
        Cleaned text with duplicated characters reduced
    """
    if not text:
        return text
    
    # Handle common patterns of character duplication
    import re
    
    # Pattern 1: Repeated characters (RRRR -> R, eeee -> e)
    # But be careful not to affect legitimate repeated characters
    cleaned = re.sub(r'([A-Za-z])\1{3,}', r'\1', text)  # 4+ repetitions -> 1
    
    # Pattern 2: Repeated punctuation (:::: -> :)
    cleaned = re.sub(r'([^\w\s])\1{2,}', r'\1', cleaned)  # 3+ repetitions -> 1
    
    return cleaned


def should_merge_numbered_section_lookahead(current_block: Dict, words: List[Dict], current_index: int) -> bool:
    """
    Look ahead to see if we should merge numbered section across gaps.
    
    This function checks if the current block is a number prefix and if the next 1-2 words
    form a heading that should be merged with it.
    
    Args:
        current_block: Current text block being processed
        words: List of all words being processed
        current_index: Index of current word in the words list
        
    Returns:
        True if numbered section should be merged with look-ahead logic
    """
    # Check if current block looks like a number prefix
    if not _is_number_prefix(current_block):
        return False
    
    # Look at next 1-2 words to see if they form a heading
    for i in range(1, min(3, len(words) - current_index)):
        next_word = words[current_index + i]
        
        # Check if this word looks like heading continuation
        if _looks_like_heading_continuation(current_block, next_word):
            return True
    
    return False


def _looks_like_heading_continuation(number_block: Dict, word: Dict) -> bool:
    """
    Check if a word looks like it continues a numbered heading.
    
    Args:
        number_block: Block containing the number prefix
        word: Word that might continue the heading
        
    Returns:
        True if word looks like heading continuation
    """
    # Must be on same page and same line
    if (number_block.get('page') != word.get('page') or
        abs(number_block.get('y0', 0) - word.get('y0', 0)) > 1.5):
        return False
    
    # Word should start with uppercase (heading-like)
    text = word.get('text', '').strip()
    if not text or not text[0].isupper():
        return False
    
    # Should be reasonably close horizontally (within 8pt gap)
    gap = word.get('x0', 0) - number_block.get('x1', 0)
    if gap < 0 or gap > 8.0:
        return False
    
    # Similar font sizes (within 2pt difference)
    number_size = number_block.get('size', 0)
    word_size = word.get('size', 0)
    if abs(number_size - word_size) > 2.0:
        return False
    
    return True


def merge_adjacent_words(words: List[Dict]) -> List[Dict]:
    """
    Merge words with identical font properties into text blocks.
    
    Merging criteria:
    - Identical font properties (fontname, size, adv)
    - Vertical alignment: abs(y0 - prev.y0) < 1.5
    - Horizontal proximity: x0 - prev.x1 ≤ 1.0
    
    Args:
        words: List of word dictionaries from a single page
        
    Returns:
        List of merged text block dictionaries
    """
    if not words:
        return []
    
    blocks = []
    current_block = None
    
    for word_index, word in enumerate(words):
        if current_block is None:
            # Start first block
            current_block = {
                'text': word['text'],
                'page': word['page'],
                'x0': word['x0'],
                'y0': word['y0'],
                'x1': word['x1'],
                'y1': word['y1'],
                'fontname': word['fontname'],
                'size': word['size'],
                'adv': word['adv']
            }
        else:
            # Calculate gap and check merging criteria
            gap = word['x0'] - current_block['x1']
            
            # Check if word can be merged with current block
            font_match = word['fontname'] == current_block['fontname'] and word['size'] == current_block['size']
            
            # Use look-ahead logic for numbered sections
            is_numbered_section = should_merge_numbered_section_lookahead(current_block, words, word_index)
            
            # Determine gap threshold - tightened thresholds for better precision
            if is_numbered_section:
                gap_threshold = max(8.0, word['size'] * 0.5)  # Tighter threshold for numbered sections
            else:
                gap_threshold = max(6.0, word['size'] * 0.4)  # Tighter threshold for regular merging
            
            can_merge = (
                (font_match or is_numbered_section) and
                # Vertical alignment - tight tolerance at 1.5pt
                abs(word['y0'] - current_block['y0']) < 1.5 and
                # Horizontal proximity - tight tolerance at 1.0pt for adjacent words
                (gap <= 1.0 or gap <= gap_threshold)
            )
            
            if can_merge:
                # Merge word into current block with proper spacing
                if gap <= 1.0:
                    current_block['text'] += word['text']  # No space for adjacent characters
                else:
                    current_block['text'] += ' ' + word['text']  # Single space for word boundaries and numbered sections
                
                # Update bounding box - ensure coordinate consistency
                old_y0, old_y1 = current_block['y0'], current_block['y1']
                new_y0 = min(current_block['y0'], word['y0'])  # y0 is top, should be minimum
                new_y1 = max(current_block['y1'], word['y1'])  # y1 is bottom, should be maximum
                
                # Validate coordinate consistency during merging
                if new_y0 > new_y1:
                    logger.error(f"Coordinate swap detected during merge: new_y0={new_y0} > new_y1={new_y1}. "
                               f"Block: '{current_block['text']}', Word: '{word['text']}'")
                
                current_block['x1'] = word['x1']
                current_block['y0'] = new_y0
                current_block['y1'] = new_y1
                
                logger.debug(f"Merged word '{word['text']}' into block. "
                           f"Coordinates: y0 {old_y0}→{new_y0}, y1 {old_y1}→{new_y1}")
                
                # Update font properties for numbered sections (use the text font, not the number font)
                if is_numbered_section:
                    current_block['fontname'] = word['fontname']  # Use the heading font
            else:
                # Finish current block and start new one
                blocks.append(current_block)
                current_block = {
                    'text': word['text'],
                    'page': word['page'],
                    'x0': word['x0'],
                    'y0': word['y0'],
                    'x1': word['x1'],
                    'y1': word['y1'],
                    'fontname': word['fontname'],
                    'size': word['size'],
                    'adv': word['adv']
                }
    
    # Don't forget the last block
    if current_block is not None:
        blocks.append(current_block)
    
    # Clean up duplicated characters in all blocks
    for block in blocks:
        block['text'] = clean_duplicated_characters(block['text'])
    
    return blocks


def merge_numbered_sections(blocks: List[Dict]) -> List[Dict]:
    """
    Post-process blocks to merge numbered sections that were split due to font differences.
    
    Args:
        blocks: List of text blocks
        
    Returns:
        List of blocks with numbered sections merged
    """
    if not blocks:
        return blocks
    
    merged_blocks = []
    i = 0
    
    while i < len(blocks):
        current_block = blocks[i]
        
        # Check if this looks like a number/bullet that should be merged with the next block
        if (i + 1 < len(blocks) and 
            _is_number_prefix(current_block) and 
            _should_merge_with_next(current_block, blocks[i + 1])):
            
            next_block = blocks[i + 1]
            
            # Merge the blocks with coordinate validation
            merged_y0 = min(current_block['y0'], next_block['y0'])  # y0 is top, should be minimum
            merged_y1 = max(current_block['y1'], next_block['y1'])  # y1 is bottom, should be maximum
            
            # Validate coordinate consistency during numbered section merging
            if merged_y0 > merged_y1:
                logger.error(f"Coordinate swap detected during numbered section merge: "
                           f"merged_y0={merged_y0} > merged_y1={merged_y1}. "
                           f"Current: '{current_block['text']}', Next: '{next_block['text']}'")
            
            merged_block = {
                'text': current_block['text'] + ' ' + next_block['text'],
                'page': current_block['page'],
                'x0': current_block['x0'],
                'y0': merged_y0,
                'x1': next_block['x1'],
                'y1': merged_y1,
                'fontname': next_block['fontname'],  # Use the text font
                'size': next_block['size'],  # Use the text size
                'adv': next_block['adv']
            }
            
            logger.debug(f"Merged numbered section: '{current_block['text']}' + '{next_block['text']}' "
                       f"→ '{merged_block['text']}'. Coordinates: y0={merged_y0}, y1={merged_y1}")
            
            merged_blocks.append(merged_block)
            i += 2  # Skip both blocks
        else:
            merged_blocks.append(current_block)
            i += 1
    
    return merged_blocks


def _is_number_prefix(block: Dict) -> bool:
    """Check if block looks like a number prefix (1., 2., etc.)."""
    text = block.get('text', '').strip()
    
    # Check for simple numbered patterns
    import re
    patterns = [
        r'^\d+\.$',  # 1., 2., etc.
        r'^[A-Z]\.$',  # A., B., etc.
        r'^[a-z]\.$',  # a., b., etc.
        r'^[IVXLCDM]+\.$',  # I., II., III., etc.
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    
    return False


def _should_merge_with_next(current_block: Dict, next_block: Dict) -> bool:
    """Check if current block should be merged with next block."""
    # Must be on same page and same line
    if (current_block.get('page') != next_block.get('page') or
        abs(current_block.get('y0', 0) - next_block.get('y0', 0)) > 2.0):
        return False
    
    # Next block should start with uppercase (heading-like)
    next_text = next_block.get('text', '').strip()
    if not next_text or not next_text[0].isupper():
        return False
    
    # Should be reasonably close horizontally - tightened threshold
    gap = next_block.get('x0', 0) - current_block.get('x1', 0)
    if gap < 0 or gap > 8:  # Tightened to 8pt gap maximum
        return False
    
    # Similar font sizes
    current_size = current_block.get('size', 0)
    next_size = next_block.get('size', 0)
    if abs(current_size - next_size) > 2.0:  # Allow 2pt difference
        return False
    
    return True


def compute_page_line_height(page_blocks: List[Dict]) -> float:
    """
    Calculate average line height for a page from block heights.
    
    Args:
        page_blocks: List of text blocks from a single page
        
    Returns:
        Average line height for the page
    """
    if not page_blocks:
        return 12.0  # Default line height
    
    line_heights = []
    for block in page_blocks:
        height = block['y1'] - block['y0']
        if height > 0:  # Only include valid heights
            line_heights.append(height)
    
    if not line_heights:
        return 12.0  # Default if no valid heights
    
    return sum(line_heights) / len(line_heights)


def detect_bold_italic(fontname: str) -> Tuple[bool, bool]:
    """
    Detect bold and italic styling from font name.
    
    Args:
        fontname: Font name string
        
    Returns:
        Tuple of (is_bold, is_italic)
    """
    fontname_lower = fontname.lower()
    
    # Common bold indicators
    is_bold = any(indicator in fontname_lower for indicator in [
        'bold', 'black', 'heavy', 'thick', 'demi', 'semi'
    ])
    
    # Common italic indicators  
    is_italic = any(indicator in fontname_lower for indicator in [
        'italic', 'oblique', 'slant'
    ])
    
    return is_bold, is_italic


def detect_number_prefix(text: str) -> bool:
    """
    Detect if text starts with numbering patterns.
    
    Patterns include:
    - Arabic numerals: 1., 1.2., 1.2.3.
    - Roman numerals: I., II., III., IV.
    - Letters: A., B., a., b.
    - Mixed: 1.A., 2.b.
    
    Args:
        text: Text content to check
        
    Returns:
        True if text starts with numbering pattern
    """
    text = text.strip()
    if not text:
        return False
    
    # Patterns for various numbering schemes
    patterns = [
        r'^\d+(\.\d+)*\.',  # 1., 1.2., 1.2.3.
        r'^[IVXLCDM]+\.',   # Roman numerals
        r'^[A-Za-z]\.',     # Single letters
        r'^\d+\.[A-Za-z]\.',  # Mixed like 1.A.
        r'^[A-Za-z]\.\d+\.',  # Mixed like A.1.
        r'^\(\d+\)',        # (1), (2)
        r'^\([A-Za-z]\)',   # (a), (b)
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    
    return False


def compute_block_features(block: Dict, prev_block: Optional[Dict], page_avg_line_height: float) -> Dict:
    """
    Add spatial and stylistic features to a text block.
    
    Features computed:
    - bold/italic detection from fontname
    - uppercase_ratio: proportion of uppercase characters
    - num_prefix: starts with numbering pattern
    - ends_with_colon: ends with colon
    - x_center: horizontal center position
    - indent: left margin position (x0)
    - gap_above: vertical gap from previous block
    - page_avg_line_height: attached for gap detection
    
    Args:
        block: Text block dictionary
        prev_block: Previous block on same page (None if first)
        page_avg_line_height: Average line height for the page
        
    Returns:
        Enhanced block dictionary with all features
    """
    # Copy original block
    enhanced_block = block.copy()
    
    # Detect bold and italic from fontname
    is_bold, is_italic = detect_bold_italic(block['fontname'])
    enhanced_block['bold'] = is_bold
    enhanced_block['italic'] = is_italic
    
    # Calculate uppercase ratio
    text = block['text']
    if text:
        uppercase_chars = sum(1 for c in text if c.isupper())
        total_letters = sum(1 for c in text if c.isalpha())
        enhanced_block['uppercase_ratio'] = uppercase_chars / total_letters if total_letters > 0 else 0.0
    else:
        enhanced_block['uppercase_ratio'] = 0.0
    
    # Detect number prefix
    enhanced_block['num_prefix'] = detect_number_prefix(text)
    
    # Check if ends with colon
    enhanced_block['ends_with_colon'] = text.strip().endswith(':')
    
    # Calculate spatial features
    enhanced_block['x_center'] = (block['x0'] + block['x1']) / 2
    enhanced_block['indent'] = block['x0']
    
    # Calculate gap above previous block
    if prev_block is not None:
        # Gap calculation: current block's top (y0) minus previous block's bottom (y1)
        gap_above = block['y0'] - prev_block['y1']
        
        # Validate gap calculation - negative gaps indicate coordinate issues
        if gap_above < -5:  # Allow small negative gaps due to overlapping text
            logger.warning(f"Negative gap detected: {gap_above:.2f}. "
                         f"Current block y0={block['y0']}, Previous block y1={prev_block['y1']}. "
                         f"Text: '{block['text'][:30]}...'")
        
        enhanced_block['gap_above'] = gap_above
        logger.debug(f"Gap above calculated: {gap_above:.2f} for block '{block['text'][:30]}...'")
    else:
        enhanced_block['gap_above'] = 0.0
    
    # Attach page average line height
    enhanced_block['page_avg_line_height'] = page_avg_line_height
    
    return enhanced_block