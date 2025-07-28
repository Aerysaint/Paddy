from typing import List, Dict, Tuple, Optional
import re
import argparse
import json
from collections import Counter
from extractor import PDFTextExtractor, TextBlock
from get_stats import FontSizeAnalyzer


def safe_print(text: str) -> None:
    """
    Print text safely, handling Unicode encoding issues.
    
    Args:
        text: Text to print
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic Unicode characters with ASCII equivalents
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)


class HumanLikeHeadingExtractor:
    def __init__(self, blocks: List[TextBlock]):
        self.blocks = [b for b in blocks if not b.is_in_table and b.text.strip()]  # Exclude table content
        self.font_analyzer = FontSizeAnalyzer(self.blocks)
        self.page_order = {id(block): i for i, block in enumerate(self.blocks)}  # Track document order
        
    def extract_title_and_headings(self) -> Tuple[Optional[TextBlock], List[Tuple[int, TextBlock]]]:
        """Extract title and headings using human-like relative analysis."""
        title = self._find_title()
        headings = self._find_headings(exclude_title=title)
        return title, headings
    
    def _find_title(self) -> Optional[TextBlock]:
        """Find document title using frequency-based approach like humans."""
        print("=== TITLE DETECTION ===")
        
        # Step 1: Check if there's a clear font size winner (least frequent, largest)
        font_freq = self._get_font_size_frequency()
        print(f"Font size frequencies: {dict(sorted(font_freq.items(), key=lambda x: x[1]))}")
        
        # Find least frequent font sizes (potential titles)
        min_frequency = min(font_freq.values())
        rare_sizes = [size for size, freq in font_freq.items() if freq == min_frequency]
        
        # If there's only one rare size and it's the largest, it's likely the title
        if len(rare_sizes) == 1 and rare_sizes[0] == max(font_freq.keys()):
            candidates = [b for b in self.blocks if round(b.font_size, 1) == rare_sizes[0]]
            print(f"Clear font size winner: {rare_sizes[0]}pt ({len(candidates)} candidates)")
            
            if len(candidates) == 1:
                print(f"Single title candidate: '{candidates[0].text}'")
                return candidates[0]
            elif len(candidates) > 1:
                # Multiple candidates with same rare large font - use other criteria
                return self._resolve_title_candidates(candidates)
        
        # Step 2: No clear font size winner - use multi-criteria scoring
        print("No clear font size winner, using multi-criteria scoring...")
        return self._find_title_by_scoring()
    
    def _resolve_title_candidates(self, candidates: List[TextBlock]) -> Optional[TextBlock]:
        """Resolve multiple title candidates using secondary criteria."""
        print(f"Resolving {len(candidates)} title candidates...")
        
        # Score each candidate
        scored_candidates = []
        for block in candidates:
            score = 0
            reasons = []
            
            # Prefer bold text
            if block.is_bold:
                score += 3
                reasons.append("bold")
            
            # Prefer centered text
            if block.horizontal_alignment == 'center':
                score += 2
                reasons.append("centered")
            
            # Prefer ALL CAPS
            if block.is_all_caps:
                score += 2
                reasons.append("all_caps")
            
            # Prefer underlined
            if block.is_underlined:
                score += 1
                reasons.append("underlined")
            
            # Prefer text that appears early in document
            early_bonus = max(0, 3 - (self.page_order[id(block)] // 5))  # Bonus for appearing in first few blocks
            score += early_bonus
            if early_bonus > 0:
                reasons.append(f"early_position({early_bonus})")
            
            # Prefer reasonable title length (not too short, not too long)
            if 10 <= block.text_length <= 100:
                score += 1
                reasons.append("good_length")
            
            scored_candidates.append((score, block, reasons))
            safe_print(f"  '{block.text[:40]}...': score={score} ({', '.join(reasons)})")
        
        # Return highest scoring candidate
        if scored_candidates:
            best = max(scored_candidates, key=lambda x: x[0])
            safe_print(f"Selected title: '{best[1].text}' (score: {best[0]})")
            return best[1]
        
        return None
    
    def _find_title_by_scoring(self) -> Optional[TextBlock]:
        """Find title when there's no clear font size winner."""
        candidates = []
        
        # Get font size statistics for relative comparison
        stats = self.font_analyzer.get_font_size_statistics()
        body_font_size = stats['mode']  # Most common = body text
        
        for block in self.blocks:
            # Skip very long text (likely paragraphs)
            if block.text_length > 200:
                continue
            
            score = 0
            reasons = []
            
            # Font size relative to body text
            if block.font_size > body_font_size:
                size_bonus = min(5, int((block.font_size - body_font_size) * 2))
                score += size_bonus
                reasons.append(f"larger_font(+{size_bonus})")
            
            # Formatting bonuses
            if block.is_bold:
                score += 3
                reasons.append("bold")
            if block.is_all_caps:
                score += 2
                reasons.append("all_caps")
            if block.is_underlined:
                score += 1
                reasons.append("underlined")
            if block.horizontal_alignment == 'center':
                score += 2
                reasons.append("centered")
            
            # Position bonus (early in document)
            position_bonus = max(0, 3 - (self.page_order[id(block)] // 3))
            score += position_bonus
            if position_bonus > 0:
                reasons.append(f"early_pos(+{position_bonus})")
            
            # Length bonus (reasonable title length)
            if 10 <= block.text_length <= 100:
                score += 1
                reasons.append("good_length")
            
            # Only consider blocks with minimum score
            if score >= 4:
                candidates.append((score, block, reasons))
        
        if not candidates:
            print("No title candidates found")
            return None
        
        # Sort by score, then by position
        candidates.sort(key=lambda x: (-x[0], self.page_order[id(x[1])]))
        
        print("Title candidates:")
        for score, block, reasons in candidates[:3]:  # Show top 3
            safe_print(f"  '{block.text[:40]}...': score={score} ({', '.join(reasons)})")
        
        winner = candidates[0]
        print(f"Selected title: '{winner[1].text}' (score: {winner[0]})")
        return winner[1]
    
    def _find_headings(self, exclude_title: Optional[TextBlock] = None) -> List[Tuple[int, TextBlock]]:
        """Find headings using human-like relative analysis."""
        print("\n=== HEADING DETECTION ===")
        
        # Get font size frequency for relative analysis
        font_freq = self._get_font_size_frequency()
        body_font_size = max(font_freq.items(), key=lambda x: x[1])[0]  # Most frequent = body
        
        print(f"Body text font size (most frequent): {body_font_size}pt")
        
        if exclude_title:
            print(f"Excluding title from headings: '{exclude_title.text[:40]}...' ({exclude_title.font_size}pt)")
        
        # Step 1: Group blocks by font size (sorted by frequency, then size)
        size_groups = self._group_blocks_by_font_size()
        
        # Step 2: Check for clear heading hierarchy by font size
        potential_heading_sizes = [size for size in size_groups.keys() if size >= body_font_size]
        potential_heading_sizes.sort(reverse=True)  # Largest first
        
        print(f"Potential heading font sizes: {potential_heading_sizes}")
        
        # Step 3: Try numbering-based hierarchy first
        numbered_headings = self._extract_numbered_headings(exclude_title)
        if numbered_headings:
            print(f"Found {len(numbered_headings)} numbered headings")
            return numbered_headings
        
        # Step 4: Use font size + formatting hierarchy
        return self._extract_font_based_headings(size_groups, body_font_size, exclude_title)
    
    def _extract_numbered_headings(self, exclude_title: Optional[TextBlock] = None) -> List[Tuple[int, TextBlock]]:
        """Extract headings based on numbering patterns with proper hierarchical structure."""
        numbered_blocks = [b for b in self.blocks if b.is_numbered and b != exclude_title]
        
        if not numbered_blocks:
            return []
        
        print(f"Found {len(numbered_blocks)} numbered blocks")
        
        # Filter out invalid heading candidates
        valid_numbered_blocks = []
        for block in numbered_blocks:
            if self._is_valid_heading_text(block):
                valid_numbered_blocks.append(block)
            else:
                safe_print(f"  Filtered out: '{block.text[:40]}...' (invalid heading text)")
        
        if not valid_numbered_blocks:
            print("No valid numbered headings found after filtering")
            return []
        
        print(f"Valid numbered blocks after filtering: {len(valid_numbered_blocks)}")
        
        # Check if we have hierarchical numbering (like 1, 1.1, 1.1.1)
        has_hierarchical_numbering = self._has_hierarchical_numbering(valid_numbered_blocks)
        
        if has_hierarchical_numbering:
            print("Detected hierarchical numbering structure - using numbering-based levels")
            return self._extract_hierarchical_numbered_headings(valid_numbered_blocks)
        else:
            print("No clear hierarchical numbering - using font size-based levels")
            return self._extract_font_size_based_numbered_headings(valid_numbered_blocks)
    
    def _has_hierarchical_numbering(self, blocks: List[TextBlock]) -> bool:
        """Check if blocks have hierarchical numbering patterns."""
        if len(blocks) < 2:
            return False
        
        # Extract all numbering information
        numbering_info = []
        
        for block in blocks:
            if not block.numbering_pattern:
                continue
            
            pattern_type, pattern_text = block.numbering_pattern.split(':', 1)
            
            # Parse different numbering types into hierarchical levels
            level = 0
            numbering = ""
            
            if pattern_type == 'decimal':
                # "1." = level 1, "2.1." = level 2, "2.1.1." = level 3
                numbering = pattern_text.rstrip('.')
                level = numbering.count('.') + 1
            elif pattern_type == 'number_space':
                # "1", "2", "3" = level 1 (main sections)
                numbering = pattern_text.strip()
                level = 1
            elif pattern_type in ['chapter', 'section', 'part_roman', 'part_decimal']:
                # Chapter/Section = level 1
                level = 1
                numbering = pattern_text.strip()
            
            if level > 0:
                numbering_info.append((level, numbering, block))
        
        if len(numbering_info) < 2:
            return False
        
        # Check if we have multiple levels
        levels = set(info[0] for info in numbering_info)
        
        # Must have at least 2 different levels and include level 1
        has_multiple_levels = len(levels) > 1
        has_level_1 = 1 in levels
        
        # Additional check: ensure we have proper parent-child relationships
        if has_multiple_levels and has_level_1:
            # Group by level
            level_groups = {}
            for level, numbering, block in numbering_info:
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append((numbering, block))
            
            # Check for logical hierarchy (level 2 items should relate to level 1 items)
            if 2 in level_groups and 1 in level_groups:
                level_2_items = [item[0] for item in level_groups[2]]
                level_1_items = [item[0] for item in level_groups[1]]
                
                # Check if level 2 items are related to level 1 items
                # e.g., "2.1", "2.2" should relate to "2"
                related_count = 0
                for l2_item in level_2_items:
                    if '.' in l2_item:
                        parent = l2_item.split('.')[0]
                        if parent in level_1_items or any(parent in l1 for l1 in level_1_items):
                            related_count += 1
                
                # If most level 2 items are related to level 1 items, it's hierarchical
                if related_count >= len(level_2_items) * 0.5:  # At least 50% related
                    return True
        
        return has_multiple_levels and has_level_1
    
    def _extract_hierarchical_numbered_headings(self, blocks: List[TextBlock]) -> List[Tuple[int, TextBlock]]:
        """Extract headings using hierarchical numbering structure."""
        headings_with_levels = []
        
        for block in blocks:
            level = self._get_hierarchical_numbering_level(block.numbering_pattern)
            if level > 0:
                headings_with_levels.append((level, block))
                safe_print(f"  Level {level}: '{block.text[:40]}...' ({block.numbering_pattern})")
        
        # Sort by document order
        headings_with_levels.sort(key=lambda x: self.page_order[id(x[1])])
        
        return headings_with_levels
    
    def _get_hierarchical_numbering_level(self, numbering_pattern: str) -> int:
        """Get heading level from hierarchical numbering pattern."""
        if not numbering_pattern or ':' not in numbering_pattern:
            return 0
        
        pattern_type, pattern_text = numbering_pattern.split(':', 1)
        
        # For decimal numbering, count the number segments to determine level
        if pattern_type == 'decimal':
            # Remove trailing dot and spaces for analysis
            numbering = pattern_text.rstrip('. ')
            if not numbering:  # Empty or just punctuation
                return 1
            
            # Count number segments separated by dots
            # "1" = 1 segment = level 1
            # "2.1" = 2 segments = level 2  
            # "3.2.1" = 3 segments = level 3
            segments = numbering.split('.')
            return len(segments)
        
        # For number_space patterns, these are main sections (level 1)
        if pattern_type == 'number_space':
            return 1
        
        # Chapter/Section/Part are level 1
        if pattern_type in ['chapter', 'section', 'part_roman', 'part_decimal']:
            return 1
        
        return 1  # Default to level 1
    
    def _extract_font_size_based_numbered_headings(self, blocks: List[TextBlock]) -> List[Tuple[int, TextBlock]]:
        """Extract headings using font size when no clear hierarchical numbering exists."""
        # Group by font size
        font_size_groups = {}
        for block in blocks:
            size = round(block.font_size, 1)
            if size not in font_size_groups:
                font_size_groups[size] = []
            font_size_groups[size].append(block)
        
        # Sort font sizes (largest first for level assignment)
        sorted_sizes = sorted(font_size_groups.keys(), reverse=True)
        
        headings_with_levels = []
        
        for level, font_size in enumerate(sorted_sizes, 1):
            blocks_at_size = font_size_groups[font_size]
            print(f"  Level {level} ({font_size}pt): {len(blocks_at_size)} headings")
            
            for block in blocks_at_size:
                headings_with_levels.append((level, block))
                safe_print(f"    '{block.text[:40]}...' ({block.numbering_pattern})")
        
        # Sort by document order
        headings_with_levels.sort(key=lambda x: self.page_order[id(x[1])])
        
        return headings_with_levels
    

    
    # def _extract_font_based_headings(self, size_groups: Dict[float, List[TextBlock]], body_font_size: float, exclude_title: Optional[TextBlock] = None) -> List[Tuple[int, TextBlock]]:
    #     """Extract headings based on font size and formatting hierarchy."""
    #     print("Using font-based heading detection...")
        
    #     # Get potential heading sizes (larger than or equal to body text)
    #     # Include body font size in case there are formatted headings at the same size
    #     heading_sizes = [size for size in size_groups.keys() if size >= body_font_size]
    #     heading_sizes.sort(reverse=True)  # Largest first
        
    #     if not heading_sizes:
    #         print("No potential heading sizes found")
    #         return []
        
    #     print(f"Heading font sizes (largest to smallest): {heading_sizes}")
        
    #     headings_with_levels = []
    #     current_level = 1
        
    #     for font_size in heading_sizes:
    #         blocks = size_groups[font_size]
            
    #         # Filter blocks that look like headings (excluding title)
    #         heading_candidates = []
    #         for block in blocks:
    #             if (block != exclude_title and 
    #                 self._is_valid_heading_text(block) and 
    #                 self._looks_like_heading(block, body_font_size)):
    #                 heading_candidates.append(block)
            
    #         if heading_candidates:
    #             print(f"Level {current_level} ({font_size}pt): {len(heading_candidates)} headings")
    #             for block in heading_candidates:
    #                 headings_with_levels.append((current_level, block))
    #                 safe_print(f"  '{block.text[:50]}...'")
                
    #             # IMPORTANT: Only increment level if we found actual headings
    #             # This ensures each distinct font size gets its own level
    #             current_level += 1
    #         else:
    #             print(f"Font size {font_size}pt: No valid heading candidates found")
        
    #     # Sort by document order
    #     headings_with_levels.sort(key=lambda x: self.page_order[id(x[1])])
        
    #     return headings_with_levels
    
    def _extract_font_based_headings(self, size_groups: Dict[float, List[TextBlock]], body_font_size: float, exclude_title: Optional[TextBlock] = None) -> List[Tuple[int, TextBlock]]:
        """
        Extract headings based on a more robust font size and formatting hierarchy.
        This revised version groups similar font sizes and establishes a clear
        hierarchy to avoid overly sensitive detection.
        """
        print("Using revised font-based heading detection...")

        # 1. Group similar font sizes together using a tolerance.
        # This prevents minor variations (e.g., 11.9pt vs 12.0pt) from creating new levels.
        font_size_clusters = {}
        sorted_sizes = sorted(size_groups.keys(), reverse=True)
        
        for size in sorted_sizes:
            # Ignore fonts smaller than the main body text.
            if size < body_font_size:
                continue
                
            is_clustered = False
            for cluster_head in font_size_clusters:
                # Group if the size is within a small tolerance (e.g., 0.5 points).
                if abs(cluster_head - size) < 0.5:
                    font_size_clusters[cluster_head].extend(size_groups[size])
                    is_clustered = True
                    break
            if not is_clustered:
                font_size_clusters[size] = size_groups[size]

        # 2. Establish a clear heading level hierarchy (H1, H2, H3, etc.).
        # The largest font size cluster is H1, the next is H2, and so on.
        cluster_heads = sorted(font_size_clusters.keys(), reverse=True)
        level_map = {cluster_head: i + 1 for i, cluster_head in enumerate(cluster_heads)}
        
        if not level_map:
            print("No potential heading sizes found after clustering.")
            return []

        print(f"Established heading levels (Font Size -> H-Level): { {k:v for k,v in level_map.items()} }")

        headings_with_levels = []
        # 3. Iterate through clusters and assign levels to valid heading candidates.
        for cluster_head, blocks in font_size_clusters.items():
            current_level = level_map[cluster_head]
            
            # Only consider up to H3 to avoid over-classifying smaller text.
            if current_level > 3:
                print(f"Skipping font size {cluster_head:.2f}pt as it exceeds H3.")
                continue

            heading_candidates = []
            for block in blocks:
                # Apply the same robust filtering logic as before.
                if (block != exclude_title and 
                    self._is_valid_heading_text(block) and 
                    self._looks_like_heading(block, body_font_size)):
                    heading_candidates.append(block)
            
            if heading_candidates:
                print(f"Level H{current_level} ({cluster_head:.2f}pt cluster): Found {len(heading_candidates)} headings.")
                for block in heading_candidates:
                    headings_with_levels.append((current_level, block))
                    safe_print(f"  -> Identified '{block.text[:60].strip()[:50]}...'")

        # 4. Sort all identified headings by their natural order in the document.
        headings_with_levels.sort(key=lambda x: self.page_order[id(x[1])])
        
        print(f"\nTotal headings extracted: {len(headings_with_levels)}")
        return headings_with_levels

    def _looks_like_heading(self, block: TextBlock, body_font_size: float) -> bool:
        """Determine if a block looks like a heading using human-like criteria."""
        # Skip very long text (likely paragraphs)
        if block.text_length > 150:
            return False
        
        # Skip very short text (likely fragments)
        if block.text_length < 5:
            return False
        
        score = 0
        
        # Font size bonus
        if block.font_size > body_font_size:
            score += 3  # Strong indicator
        elif block.font_size == body_font_size:
            score += 1  # Same size but might be formatted differently
        
        # Formatting bonuses
        if block.is_bold:
            score += 3  # Strong heading indicator
        if block.is_all_caps:
            score += 2
        if block.is_underlined:
            score += 2
        
        # Spacing bonus (headings typically have more space around them)
        if block.space_above > 15:
            score += 2
        elif block.space_above > 8:
            score += 1
        
        if block.space_below > 15:
            score += 2
        elif block.space_below > 8:
            score += 1
        
        # Numbering bonus
        if block.is_numbered:
            score += 3  # Strong heading indicator
        
        # Alignment bonus
        if block.horizontal_alignment == 'center':
            score += 2
        
        # Length bonus (headings are typically concise)
        if 10 <= block.text_length <= 80:
            score += 1
        
        # Need minimum score to be considered a heading
        # Lower threshold for same-size text with strong formatting
        min_score = 2 if block.font_size == body_font_size else 3
        return score >= min_score
    
    def _get_font_size_frequency(self) -> Dict[float, int]:
        """Get frequency of each font size (rounded to 1 decimal)."""
        sizes = [round(block.font_size, 1) for block in self.blocks]
        return dict(Counter(sizes))
    
    def _group_blocks_by_font_size(self) -> Dict[float, List[TextBlock]]:
        """Group blocks by font size."""
        groups = {}
        for block in self.blocks:
            size = round(block.font_size, 1)
            if size not in groups:
                groups[size] = []
            groups[size].append(block)
        return groups
    
    def _is_valid_heading_text(self, block: TextBlock) -> bool:
        """Check if text is valid for a heading (not purely numeric or special characters)."""
        text = block.text.strip()
        
        if not text:
            return False
        
        # Don't reject text that starts with numbers if it has meaningful content after
        # This was the main bug - rejecting "1 Introduction", "2 Background", etc.
        
        # Remove common heading prefixes/suffixes for analysis
        clean_text = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', text)  # Remove leading numbers like "1. ", "3.1 ", "3.2.1 "
        clean_text = re.sub(r'\s*\d+\.?\s*$', '', clean_text)  # Remove trailing numbers
        clean_text = clean_text.strip()
        
        if not clean_text:
            return False
        
        # Check if remaining text is purely numeric
        if re.match(r'^[\d\s\.\-\+\(\)]+$', clean_text):
            return False
        
        # Check if remaining text is purely special characters/punctuation
        if re.match(r'^[\W\s]+$', clean_text):
            return False
        
        # Must contain at least some alphabetic characters
        if not re.search(r'[a-zA-Z]', clean_text):
            return False
        
        # Must have reasonable length after cleaning
        if len(clean_text) < 3:
            return False
        
        # Check for obvious non-heading patterns (but be more specific)
        non_heading_patterns = [
            r'^\d+\s*$',  # Just numbers
            r'^[\d\s\.\-\+\(\)]+$',  # Only numbers and math symbols
            r'^[^\w\s]+$',  # Only special characters
            r'^\d+\s+\d+\s*$',  # Two numbers only
            r'^\d+\s+\d+\s+\d+\s*$',  # Three numbers only
            r'^\d+\.\d+\s*$',  # Decimal numbers only
            r'^[\•\-\–\—\*]\s*$',  # Just bullet points or dashes
        ]
        
        for pattern in non_heading_patterns:
            if re.match(pattern, text):
                return False
        
        # Additional check: reject obvious bullet points or list items
        # But allow numbered headings like "1 Introduction"
        if re.match(r'^[\•\-\–\—\*]\s', text):
            return False
        
        # Allow numbered headings but reject pure mathematical expressions
        # This allows "1 Introduction" but rejects "1 + 2 = 3"
        if re.match(r'^\d+\s+[a-zA-Z]', text):
            return True  # Explicitly allow numbered headings
        
        return True
    



    def print_results(self, title: Optional[TextBlock], headings: List[Tuple[int, TextBlock]]):
        """Print the extracted title and heading hierarchy."""
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        if title:
            print(f"TITLE: {title.text}")
            print(f"       Font: {title.font_size:.1f}pt, Bold: {title.is_bold}, "
                  f"Caps: {title.is_all_caps}, Align: {title.horizontal_alignment}")
            print()
        else:
            print("TITLE: None detected")
            print()
        
        if headings:
            print("HEADINGS:")
            for level, block in headings:
                indent = "  " * (level - 1)
                style_info = []
                if block.is_bold:
                    style_info.append("BOLD")
                if block.is_all_caps:
                    style_info.append("CAPS")
                if block.is_underlined:
                    style_info.append("UNDERLINED")
                
                style_str = f" [{', '.join(style_info)}]" if style_info else ""
                numbering_str = f" ({block.numbering_pattern})" if block.is_numbered else ""
                
                safe_print(f"{indent}H{level}: {block.text}")
                print(f"{indent}     Font: {block.font_size:.1f}pt{style_str}{numbering_str}")
        else:
            print("HEADINGS: None detected")
    
    def to_json(self, title: Optional[TextBlock], headings: List[Tuple[int, TextBlock]]) -> Dict:
        """Convert title and headings to JSON format matching the schema."""
        # Extract title text
        title_text = title.text.strip() if title else "Untitled Document"
        
        # Build outline array
        outline = []
        for level, block in headings:
            outline.append({
                "level": f"H{level}",
                "text": block.text.strip(),
                "page": block.page_number
            })
        
        return {
            "title": title_text,
            "outline": outline
        }


def main():
    parser = argparse.ArgumentParser(description="Extract title and headings using human-like analysis")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    parser.add_argument("--format", choices=['json', 'text'], default='json',
                       help="Output format: json (default) or text")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed analysis process")
    
    args = parser.parse_args()
    
    # Extract text blocks
    extractor = PDFTextExtractor()
    blocks = extractor.extract_text_blocks(args.pdf_path)
    
    if not blocks:
        print("No text blocks found in the PDF.")
        return
    
    # Extract title and headings using human-like approach
    heading_extractor = HumanLikeHeadingExtractor(blocks)
    title, headings = heading_extractor.extract_title_and_headings()
    
    if args.format == 'json':
        # Generate JSON output
        result = heading_extractor.to_json(title, headings)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"JSON output saved to: {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Print results in text format
        heading_extractor.print_results(title, headings)
    
    return title, headings


if __name__ == "__main__":
    main()