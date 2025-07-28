import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import argparse
import json
import re


@dataclass
class TextBlock:
    """Represents a contiguous block of text with consistent formatting."""
    text: str
    space_above: float
    space_below: float
    is_bold: bool
    is_italic: bool
    is_underlined: bool
    is_in_table: bool
    font_size: float
    font_face: str
    font_color: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_number: int  # Page number where this text block appears (1-indexed)
    # Additional parameters for heading detection
    is_all_caps: bool  # ALL CAPS text often indicates headings
    text_length: int  # Short text blocks are more likely to be headings
    line_count: int  # Number of lines in the block
    horizontal_alignment: str  # 'left', 'center', 'right', 'justified'
    indentation_level: float  # Distance from left margin
    is_numbered: bool  # Contains numbering like "1.", "1.1", "Chapter 1"
    numbering_pattern: str  # The actual numbering pattern found


class PDFTextExtractor:
    def __init__(self):
        self.tolerance = 2.0  # Tolerance for grouping similar formatting
    
    def extract_text_blocks(self, pdf_path: str) -> List[TextBlock]:
        """Extract text blocks from PDF, grouping contiguous text with similar formatting."""
        doc = fitz.open(pdf_path)
        all_blocks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_blocks = self._extract_page_blocks(page, page_num + 1)  # 1-indexed page numbers
            all_blocks.extend(page_blocks)
        
        doc.close()
        return all_blocks
    
    def _extract_page_blocks(self, page, page_number: int) -> List[TextBlock]:
        """Extract and group text blocks from a single page."""
        # Get text with detailed formatting information
        text_dict = page.get_text("dict")
        
        all_spans = []
        
        # First, collect all text spans from all blocks and lines
        for block in text_dict["blocks"]:
            if "lines" not in block:  # Skip image blocks
                continue
            
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():  # Only include non-empty spans
                        all_spans.append(span)
        
        # Sort spans by vertical position (top to bottom), then horizontal (left to right)
        all_spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        
        # Process spans and calculate spacing
        processed_spans = []
        for i, span in enumerate(all_spans):
            # Calculate space above from previous span
            space_above = 0
            if i > 0:
                prev_span = all_spans[i-1]
                space_above = max(0, span["bbox"][1] - prev_span["bbox"][3])
            
            processed_span = self._process_span(span, space_above)
            processed_spans.append(processed_span)
        
        # Group spans with similar formatting that should be one text block
        grouped_blocks = self._group_similar_blocks(processed_spans, page_number)
        
        # Use PyMuPDF's built-in table detection
        self._detect_table_blocks_with_pymupdf(page, grouped_blocks)
        
        return grouped_blocks
    
    def _process_span(self, span, space_above: float) -> dict:
        """Process a single span and extract formatting information."""
        # Extract formatting information
        flags = span["flags"]
        is_bold = bool(flags & 2**4)
        is_italic = bool(flags & 2**1)
        is_underlined = bool(flags & 2**0)
        
        # Convert color from integer to hex
        color = span["color"]
        font_color = f"#{color:06x}" if color != 0 else "#000000"
        
        return {
            "text": span["text"],
            "bbox": span["bbox"],
            "space_above": space_above,
            "is_bold": is_bold,
            "is_italic": is_italic,
            "is_underlined": is_underlined,
            "font_size": span["size"],
            "font_face": span["font"],
            "font_color": font_color,
            "flags": flags
        }
    
    def _group_similar_blocks(self, blocks: List[dict], page_number: int) -> List[TextBlock]:
        """Group consecutive blocks with similar formatting into single text blocks."""
        if not blocks:
            return []
        
        grouped_blocks = []
        current_group = [blocks[0]]
        
        for i in range(1, len(blocks)):
            current_block = blocks[i]
            previous_block = blocks[i-1]
            
            # Check if blocks should be grouped together
            if self._should_group_blocks(previous_block, current_block):
                current_group.append(current_block)
            else:
                # Finalize current group and start new one
                grouped_blocks.append(self._create_text_block(current_group, page_number))
                current_group = [current_block]
        
        # Don't forget the last group
        if current_group:
            grouped_blocks.append(self._create_text_block(current_group, page_number))
        
        return grouped_blocks
    
    def _should_group_blocks(self, block1: dict, block2: dict) -> bool:
        """Determine if two blocks should be grouped together."""
        # Check formatting similarity
        formatting_match = (
            block1["is_bold"] == block2["is_bold"] and
            block1["is_italic"] == block2["is_italic"] and
            block1["is_underlined"] == block2["is_underlined"] and
            abs(block1["font_size"] - block2["font_size"]) < self.tolerance and
            block1["font_face"] == block2["font_face"] and
            block1["font_color"] == block2["font_color"]
        )
        
        if not formatting_match:
            return False
        
        # Calculate spacing between blocks
        vertical_gap = block2["bbox"][1] - block1["bbox"][3]  # Gap between bottom of block1 and top of block2
        horizontal_distance = abs(block1["bbox"][0] - block2["bbox"][0])  # Horizontal alignment difference
        
        # Determine if blocks should be grouped based on spacing
        line_height = block1["font_size"] * 1.2  # Approximate line height
        
        # Group if:
        # 1. They're on the same line (very small vertical gap)
        # 2. They're on consecutive lines with normal line spacing and similar horizontal alignment
        # 3. The vertical gap is small enough to be considered part of the same paragraph
        
        same_line = vertical_gap < self.tolerance
        consecutive_lines = vertical_gap < line_height * 1.5 and horizontal_distance < block1["font_size"]
        paragraph_continuation = vertical_gap < line_height * 2.5  # Allow for slightly larger gaps within paragraphs
        
        # For paragraph continuation, also check if there's reasonable text flow
        # (not a huge horizontal jump that would indicate a new column or section)
        reasonable_flow = horizontal_distance < block1["font_size"] * 10
        
        return same_line or (consecutive_lines and reasonable_flow) or (paragraph_continuation and reasonable_flow)
    
    def _create_text_block(self, group: List[dict], page_number: int) -> TextBlock:
        """Create a TextBlock from a group of similar blocks."""
        if not group:
            return None
        
        # Intelligently combine text from all blocks in group
        combined_text = self._combine_text_intelligently(group)
        
        # Use formatting from first block (they should all be similar)
        first_block = group[0]
        last_block = group[-1]
        
        # Calculate space above and below
        space_above = first_block["space_above"]
        
        # Space below is harder to calculate without knowing the next block
        # For now, we'll set it to 0 and calculate it in post-processing
        space_below = 0
        
        # Table detection will be done later at the page level
        is_in_table = False
        
        # Calculate combined bounding box
        min_x = min(block["bbox"][0] for block in group)
        min_y = min(block["bbox"][1] for block in group)
        max_x = max(block["bbox"][2] for block in group)
        max_y = max(block["bbox"][3] for block in group)
        combined_bbox = (min_x, min_y, max_x, max_y)
        
        # Calculate additional heading-related parameters
        text_stripped = combined_text.strip()
        is_all_caps = self._is_all_caps_unicode_safe(text_stripped)
        text_length = len(text_stripped)
        line_count = len([line for line in text_stripped.split('\n') if line.strip()])
        
        # Calculate horizontal alignment (simplified)
        page_width = 595  # Approximate A4 width in points
        text_center = (combined_bbox[0] + combined_bbox[2]) / 2
        page_center = page_width / 2
        
        if abs(text_center - page_center) < 50:  # Within 50 points of center
            horizontal_alignment = 'center'
        elif combined_bbox[0] < 100:  # Close to left margin
            horizontal_alignment = 'left'
        elif combined_bbox[2] > page_width - 100:  # Close to right margin
            horizontal_alignment = 'right'
        else:
            horizontal_alignment = 'left'  # Default
        
        # Calculate indentation level (distance from left margin)
        indentation_level = combined_bbox[0]
        
        # Detect numbering patterns
        is_numbered, numbering_pattern = self._detect_numbering_pattern(text_stripped)
        
        return TextBlock(
            text=text_stripped,
            space_above=space_above,
            space_below=space_below,
            is_bold=first_block["is_bold"],
            is_italic=first_block["is_italic"],
            is_underlined=first_block["is_underlined"],
            is_in_table=is_in_table,
            font_size=first_block["font_size"],
            font_face=first_block["font_face"],
            font_color=first_block["font_color"],
            bbox=combined_bbox,
            page_number=page_number,
            is_all_caps=is_all_caps,
            text_length=text_length,
            line_count=line_count,
            horizontal_alignment=horizontal_alignment,
            indentation_level=indentation_level,
            is_numbered=is_numbered,
            numbering_pattern=numbering_pattern
        )
    
    def _combine_text_intelligently(self, group: List[dict]) -> str:
        """Combine text from spans, handling line breaks intelligently."""
        if not group:
            return ""
        
        if len(group) == 1:
            return group[0]["text"]
        
        combined_parts = []
        
        for i, block in enumerate(group):
            text = block["text"]
            
            if i == 0:
                combined_parts.append(text)
            else:
                prev_block = group[i-1]
                
                # Check if we need to add space between blocks
                # If the previous text ends with a hyphen, it might be a word break
                if prev_block["text"].rstrip().endswith('-'):
                    # Remove the hyphen and join without space (word continuation)
                    if combined_parts:
                        combined_parts[-1] = combined_parts[-1].rstrip()[:-1]  # Remove the hyphen
                    combined_parts.append(text.lstrip())
                elif prev_block["text"].rstrip().endswith((' ', '\t')):
                    # Previous text already has trailing space
                    combined_parts.append(text.lstrip())
                elif text.startswith((' ', '\t')):
                    # Current text has leading space
                    combined_parts.append(text)
                else:
                    # Add space between words
                    combined_parts.append(' ' + text.lstrip())
        
        return ''.join(combined_parts)
    
    def _detect_numbering_pattern(self, text: str) -> Tuple[bool, str]:
        """Detect if text contains heading numbering patterns."""
        import re
        
        text = text.strip()
        
        # Common heading numbering patterns
        patterns = [
            (r'^\d+(?:\.\d+)*\s+', 'decimal'),  # 1 Introduction, 2.1 Audience, 3.2.1 Subsection
            (r'^\d+(?:\.\d+)*\.\s*', 'decimal'),  # 1., 1.1., 1.1.1., 2.1., etc.
            (r'^[IVX]+\.\s*', 'roman_upper'),  # I., II., III.
            (r'^[ivx]+\.\s*', 'roman_lower'),  # i., ii., iii.
            (r'^[A-Z]\.\s*', 'alpha_upper'),  # A., B., C.
            (r'^[a-z]\.\s*', 'alpha_lower'),  # a., b., c.
            (r'^[A-Z]\)\s*', 'alpha_upper_paren'),  # A), B), C)
            (r'^[a-z]\)\s*', 'alpha_lower_paren'),  # a), b), c)
            (r'^\d+(?:\.\d+)*\)\s*', 'decimal_paren'),  # 1), 2.1), 3.1.1), etc.
            (r'^Chapter\s+\d+', 'chapter'),  # Chapter 1, Chapter 2
            (r'^Section\s+\d+', 'section'),  # Section 1, Section 2
            (r'^Part\s+[IVX]+', 'part_roman'),  # Part I, Part II
            (r'^Part\s+\d+', 'part_decimal'),  # Part 1, Part 2
        ]
        
        for pattern, pattern_type in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return True, f"{pattern_type}:{match.group().strip()}"
        
        return False, ""
    
    def _is_all_caps_unicode_safe(self, text: str) -> bool:
        """Check if text is ALL CAPS, handling Unicode characters safely."""
        if not text or not text.strip():
            return False
        
        # Remove whitespace and punctuation for analysis
        import unicodedata
        
        # Get only letter characters (handles Unicode properly)
        letters_only = ''.join(char for char in text if unicodedata.category(char).startswith('L'))
        
        if not letters_only or len(letters_only) < 3:
            return False  # No letters found or too few letters
        
        # Check if all letters are uppercase
        # This handles Unicode characters properly including Greek, accented chars, etc.
        return letters_only.isupper()
    
    def _detect_table_blocks_with_pymupdf(self, page, blocks: List[TextBlock]) -> None:
        """Use PyMuPDF's built-in table detection to identify table regions."""
        try:
            # Use PyMuPDF's find_tables method
            tables = page.find_tables()
            
            # Create list of table bounding boxes
            table_regions = []
            for table in tables:
                # Get the table's bounding box
                table_bbox = table.bbox
                table_regions.append(table_bbox)
            
            # Mark blocks that fall within any table region, but with additional validation
            for block in blocks:
                if self._is_block_in_table_region(block, table_regions):
                    # Additional validation: check if this block is likely NOT a title/heading
                    if not self._is_likely_title_or_heading(block, blocks):
                        block.is_in_table = True
                    else:
                        block.is_in_table = False
                else:
                    block.is_in_table = False
                
        except Exception as e:
            # Fallback to heuristic method if PyMuPDF table detection fails
            print(f"PyMuPDF table detection failed: {e}")
            self._detect_table_blocks_fallback(blocks)
    
    def _detect_table_blocks_fallback(self, blocks: List[TextBlock]) -> None:
        """Fallback heuristic table detection method."""
        # Simple fallback - mark blocks with obvious table indicators
        for block in blocks:
            text = block.text.lower()
            
            # Basic table indicators
            table_indicators = [
                's.no' in text and 'name' in text,  # Table headers
                any(f'{i}.' == block.text.strip() for i in range(1, 20)),  # Numbered items
                'rs.' in text and len(text.split()) <= 5,  # Currency amounts
            ]
            
            # Exclude obvious non-table content
            exclude_patterns = [
                'application form', 'signature of', 'government servant',
                'i declare', 'particulars furnished', 'undertake to refund'
            ]
            
            is_excluded = any(pattern in text for pattern in exclude_patterns)
            is_table_content = any(table_indicators) and not is_excluded
            
            block.is_in_table = is_table_content
    
    def _is_block_in_table_region(self, block: TextBlock, table_regions: List[Tuple[float, float, float, float]]) -> bool:
        """Check if a block falls within any table region."""
        block_x, block_y, block_x2, block_y2 = block.bbox
        
        for region_x, region_y, region_x2, region_y2 in table_regions:
            # Check if block overlaps with this region
            if not (block_x2 < region_x or block_x > region_x2 or 
                   block_y2 < region_y or block_y > region_y2):
                return True
        
        return False
    
    def _is_likely_title_or_heading(self, block: TextBlock, all_blocks: List[TextBlock]) -> bool:
        """Check if a block is likely a title or heading using structural analysis, not keywords."""
        
        # Calculate font size statistics for relative comparison
        font_sizes = [b.font_size for b in all_blocks if b.text.strip()]
        if not font_sizes:
            return False
        
        avg_font_size = sum(font_sizes) / len(font_sizes)
        max_font_size = max(font_sizes)
        min_font_size = min(font_sizes)
        
        # Get block position in document
        block_position = 0
        for i, b in enumerate(all_blocks):
            if b == block:
                block_position = i
                break
        
        score = 0
        
        # 1. Font size analysis (strongest indicator)
        font_size_percentile = (block.font_size - min_font_size) / (max_font_size - min_font_size) if max_font_size > min_font_size else 0
        
        if font_size_percentile >= 0.8:  # Top 20% of font sizes
            score += 4
        elif font_size_percentile >= 0.6:  # Top 40% of font sizes
            score += 2
        elif font_size_percentile >= 0.4:  # Above average
            score += 1
        
        # 2. Position in document (titles/headings appear early)
        position_percentile = block_position / len(all_blocks) if len(all_blocks) > 1 else 0
        
        if position_percentile <= 0.1:  # First 10% of document
            score += 3
        elif position_percentile <= 0.3:  # First 30% of document
            score += 2
        elif position_percentile <= 0.5:  # First half of document
            score += 1
        
        # 3. Formatting indicators
        if block.is_bold:
            score += 2
        if block.is_all_caps:
            score += 2
        if block.is_underlined:
            score += 1
        
        # 4. Layout indicators
        if block.horizontal_alignment == 'center':
            score += 2
        
        # 5. Spacing analysis (titles/headings have more whitespace)
        avg_spacing_above = sum(b.space_above for b in all_blocks) / len(all_blocks)
        avg_spacing_below = sum(b.space_below for b in all_blocks) / len(all_blocks)
        
        if block.space_above > avg_spacing_above * 1.5:
            score += 1
        if block.space_below > avg_spacing_below * 1.5:
            score += 1
        
        # 6. Text length analysis (titles/headings are typically concise)
        if 5 <= block.text_length <= 100:
            score += 1
        elif block.text_length > 200:  # Very long text unlikely to be title
            score -= 2
        
        # 7. Structural patterns that indicate table content (language-agnostic)
        import re
        text = block.text.strip()
        
        # Multiple short words pattern (common in table headers)
        words = text.split()
        if len(words) >= 3:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length <= 6 and len(words) >= 4:  # Many short words
                score -= 3
        
        # Patterns that suggest tabular data
        tabular_patterns = [
            r'^\d+\.?\s+\w+\s+\w+\s+\w+',  # Number followed by multiple words (like "1. Name Age Relationship")
            r'^\w+\s+\w+\s+\w+\s+\w+\s*$',  # 4+ single words in sequence
            r'.*\d+.*\d+.*',  # Contains multiple numbers (common in table headers/data)
        ]
        
        for pattern in tabular_patterns:
            if re.match(pattern, text):
                score -= 2
        
        # 8. Context analysis - compare with surrounding blocks
        surrounding_blocks = []
        for i in range(max(0, block_position - 2), min(len(all_blocks), block_position + 3)):
            if i != block_position:
                surrounding_blocks.append(all_blocks[i])
        
        if surrounding_blocks:
            # If this block is much larger than surrounding blocks, likely a heading
            surrounding_font_sizes = [b.font_size for b in surrounding_blocks]
            avg_surrounding_size = sum(surrounding_font_sizes) / len(surrounding_font_sizes)
            
            if block.font_size > avg_surrounding_size * 1.3:
                score += 2
            
            # If this block has similar formatting to many surrounding blocks, less likely to be special
            similar_formatting_count = 0
            for b in surrounding_blocks:
                if (b.is_bold == block.is_bold and 
                    abs(b.font_size - block.font_size) < 1 and
                    b.horizontal_alignment == block.horizontal_alignment):
                    similar_formatting_count += 1
            
            if similar_formatting_count >= len(surrounding_blocks) * 0.7:  # 70% similar
                score -= 1
        
        # 9. Frequency analysis - if this formatting is rare in document, more likely to be special
        similar_blocks = [b for b in all_blocks if 
                         abs(b.font_size - block.font_size) < 1 and 
                         b.is_bold == block.is_bold]
        
        rarity_score = 1 - (len(similar_blocks) / len(all_blocks))
        if rarity_score > 0.8:  # Very rare formatting
            score += 2
        elif rarity_score > 0.6:  # Somewhat rare
            score += 1
        
        # Threshold: if score is high enough, it's likely a title/heading
        return score >= 6
    



def main():
    parser = argparse.ArgumentParser(description="Extract formatted text blocks from PDF")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output file (optional)")
    
    args = parser.parse_args()
    
    extractor = PDFTextExtractor()
    text_blocks = extractor.extract_text_blocks(args.pdf_path)
    
    # Post-process to calculate space_below
    for i in range(len(text_blocks) - 1):
        # This is a simplified calculation - in reality, you'd need more sophisticated spacing detection
        text_blocks[i].space_below = text_blocks[i+1].space_above
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, block in enumerate(text_blocks):
                f.write(f"Block {i+1}:\n")
                f.write(f"Text: {block.text}\n")
                f.write(f"Page: {block.page_number}\n")
                f.write(f"Space Above: {block.space_above:.2f}\n")
                f.write(f"Space Below: {block.space_below:.2f}\n")
                f.write(f"Bold: {block.is_bold}\n")
                f.write(f"Italic: {block.is_italic}\n")
                f.write(f"Underlined: {block.is_underlined}\n")
                f.write(f"In Table: {block.is_in_table}\n")
                f.write(f"Font Size: {block.font_size}\n")
                f.write(f"Font Face: {block.font_face}\n")
                f.write(f"Font Color: {block.font_color}\n")
                f.write("-" * 50 + "\n")
    else:
        for i, block in enumerate(text_blocks):
            print(f"Block {i+1}: {block.text[:50]}...")
            print(f"  Page: {block.page_number}")
            print(f"  Format: {block.font_face}, {block.font_size}pt, {block.font_color}")
            print(f"  Style: Bold={block.is_bold}, Italic={block.is_italic}, Underlined={block.is_underlined}")
            print(f"  Spacing: Above={block.space_above:.1f}, Below={block.space_below:.1f}")
            print(f"  Table: {block.is_in_table}")
            print()
    
    return text_blocks


if __name__ == "__main__":
    main()