from typing import List, Dict, Any, Optional
import json
import os
from extractor import PDFTextExtractor, TextBlock

def _merge_small_chunks(chunks: List[Dict[str, Any]], min_length: int = 250) -> List[Dict[str, Any]]:
    """
    Post-processes a list of chunks to merge small, insignificant chunks
    into the previous one. This helps clean up noise from overly sensitive
    heading extraction by ensuring a minimum chunk size.
    
    Args:
        chunks: The list of chunks to process.
        min_length: The minimum character length for a chunk to be considered standalone.
        
    Returns:
        A new list of chunks with small fragments merged.
    """
    if len(chunks) < 2:
        return chunks

    print(f"\nRunning final merge pass to clean up small chunks (min length: {min_length})...")
    merged_chunks = []
    # Start with the first chunk, as it cannot be merged into a previous one.
    current_chunk = chunks[0]

    for next_chunk in chunks[1:]:
        # Check if the *next* chunk is too small to stand on its own.
        if len(next_chunk['text_content']) < min_length:
            # Merge the small 'next_chunk' into the 'current_chunk'.
            # We can format it like a sub-section for clarity.
            current_chunk['text_content'] += f"\n\n---\n\n**{next_chunk['heading_text']}**\n{next_chunk['text_content']}"
            safe_print(f"  -> Merging small chunk '{next_chunk['heading_text'][:50]}...'")
        else:
            # The next chunk is large enough. Finalize the current_chunk and
            # make the next_chunk the new current_chunk.
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk

    # Don't forget to add the last processed chunk.
    merged_chunks.append(current_chunk)
    
    print(f"Merge pass complete. Reduced from {len(chunks)} to {len(merged_chunks)} chunks.")
    return merged_chunks

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


def create_chunks(structured_content: Dict[str, Any], pdf_filename: str) -> List[Dict[str, Any]]:
    """
    Create chunks from structured content, with intelligent content aggregation.
    """
    # First create individual chunks as before
    individual_chunks = _create_individual_chunks(structured_content, pdf_filename)
    
    # Then aggregate related chunks into comprehensive content blocks
    aggregated_chunks = _aggregate_related_chunks(individual_chunks)
    final_chunks = _merge_small_chunks(aggregated_chunks)
    return final_chunks


def _create_individual_chunks(structured_content: Dict[str, Any], pdf_filename: str) -> List[Dict[str, Any]]:
    """
    Create individual chunks from structured content extracted by heading_extractor.py.
    
    Args:
        structured_content: JSON output from heading_extractor.py containing title and outline
        pdf_filename: Name of the source PDF file
        
    Returns:
        List of individual chunk dictionaries
    """
    if not structured_content or 'outline' not in structured_content:
        return []
    
    # Extract all text blocks from the PDF to get the actual content
    extractor = PDFTextExtractor()
    all_blocks = extractor.extract_text_blocks(pdf_filename)
    
    if not all_blocks:
        print(f"Warning: Could not extract text blocks from {pdf_filename}")
        return _create_chunks_from_headings_only(structured_content, pdf_filename)
    
    outline = structured_content.get('outline', [])
    title = structured_content.get('title', 'Untitled Document')
    
    chunks = []
    
    # Add title as first chunk if it exists and is meaningful
    if title and title.strip() and title != 'Untitled Document':
        chunks.append({
            'text_content': title.strip(),
            'source_document': pdf_filename,
            'page_number': 1,
            'heading_level': 'TITLE',
            'heading_text': title.strip()
        })
    
    # Create a mapping of heading text to blocks for faster lookup
    heading_to_block = {}
    for block in all_blocks:
        if block.text.strip():
            heading_to_block[block.text.strip()] = block
    
    # Process outline entries
    for i, heading in enumerate(outline):
        level = heading.get('level', 'H1')
        text = heading.get('text', '').strip()
        page = heading.get('page', 1)
        
        if not text:
            continue
        
        # Skip table of contents entries
        if _is_table_of_contents_entry(text):
            continue
        
        # For H1 and H2 headings, create chunks with full content
        if level in ['H1', 'H2']:
            chunk_content = _extract_content_for_heading(
                text, i, outline, all_blocks, heading_to_block
            )
            
            chunks.append({
                'text_content': chunk_content,
                'source_document': pdf_filename,
                'page_number': page,
                'heading_level': level,
                'heading_text': text
            })
    
    return chunks


def _aggregate_related_chunks(individual_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate related individual chunks into comprehensive content blocks.
    
    This function combines micro-headings and fragments into meaningful sections
    that provide comprehensive information about topics.
    
    Args:
        individual_chunks: List of individual chunks from heading extraction
        
    Returns:
        List of aggregated chunks with comprehensive content
    """
    if not individual_chunks:
        return []
    
    aggregated_chunks = []
    current_section = None
    current_content_parts = []
    
    for chunk in individual_chunks:
        heading_text = chunk['heading_text']
        text_content = chunk['text_content']
        
        # Skip very short fragments that are likely not real headings
        if len(heading_text.strip()) < 3:
            continue
        
        # Check if this looks like a main section heading
        if _is_main_section_heading(heading_text, text_content):
            # Save previous section if it exists
            if current_section and current_content_parts:
                current_section['text_content'] = _combine_content_parts(current_content_parts)
                aggregated_chunks.append(current_section)
            
            # Start new section
            current_section = chunk.copy()
            current_content_parts = [text_content]
        
        # Check if this is content that should be added to current section
        elif current_section and _should_aggregate_with_current_section(heading_text, text_content, current_section):
            # Add to current section
            current_content_parts.append(f"{heading_text}: {text_content}" if heading_text != text_content else text_content)
        
        # This might be a standalone section
        else:
            # Save previous section if it exists
            if current_section and current_content_parts:
                current_section['text_content'] = _combine_content_parts(current_content_parts)
                aggregated_chunks.append(current_section)
            
            # Create standalone section
            current_section = chunk.copy()
            current_content_parts = [text_content]
    
    # Don't forget the last section
    if current_section and current_content_parts:
        current_section['text_content'] = _combine_content_parts(current_content_parts)
        aggregated_chunks.append(current_section)
    
    return aggregated_chunks


def _is_main_section_heading(heading_text: str, text_content: str) -> bool:
    """
    Determine if a heading represents a main section that should start a new aggregated chunk.
    
    Args:
        heading_text: The heading text
        text_content: The content text
        
    Returns:
        True if this should start a new section
    """
    # If heading and content are the same and it's substantial, it's likely a main heading
    if heading_text == text_content and len(heading_text.strip()) > 10:
        return True
    
    # Look for common main section patterns
    main_section_keywords = [
        'activities', 'adventures', 'experiences', 'attractions', 'things to do',
        'restaurants', 'hotels', 'accommodation', 'dining', 'cuisine', 'food',
        'nightlife', 'entertainment', 'shopping', 'markets', 'culture', 'history',
        'museums', 'art', 'festivals', 'events', 'transportation', 'travel tips',
        'best time', 'weather', 'climate', 'packing', 'tips and tricks',
        'outdoor', 'sports', 'hiking', 'beaches', 'coastal', 'water sports',
        'wine', 'vineyards', 'cooking', 'classes', 'tours', 'wellness', 'spa'
    ]
    
    heading_lower = heading_text.lower()
    for keyword in main_section_keywords:
        if keyword in heading_lower:
            return True
    
    return False


def _should_aggregate_with_current_section(heading_text: str, text_content: str, current_section: Dict[str, Any]) -> bool:
    """
    Determine if content should be aggregated with the current section.
    
    Args:
        heading_text: The heading text to consider
        text_content: The content text
        current_section: The current section being built
        
    Returns:
        True if this content should be added to the current section
    """
    if not current_section:
        return False
    
    current_heading = current_section['heading_text'].lower()
    new_heading = heading_text.lower()
    
    # If the new heading starts with ":" it's likely a continuation
    if heading_text.startswith(':'):
        return True
    
    # If it's a very short fragment, it might be part of a list
    if len(heading_text.strip()) < 20 and ':' not in heading_text:
        return True
    
    # Check for thematic similarity
    if 'beach' in current_heading and any(word in new_heading for word in ['beach', 'coast', 'sea', 'water', 'swimming']):
        return True
    
    if 'food' in current_heading or 'cuisine' in current_heading:
        if any(word in new_heading for word in ['restaurant', 'dish', 'wine', 'cooking', 'food', 'eat']):
            return True
    
    if 'hotel' in current_heading or 'accommodation' in current_heading:
        if any(word in new_heading for word in ['hotel', 'stay', 'room', 'accommodation']):
            return True
    
    return False


def _combine_content_parts(content_parts: List[str]) -> str:
    """
    Combine multiple content parts into a coherent text block.
    
    Args:
        content_parts: List of content strings to combine
        
    Returns:
        Combined content string
    """
    if not content_parts:
        return ""
    
    # Remove duplicates while preserving order
    seen = set()
    unique_parts = []
    for part in content_parts:
        part_clean = part.strip()
        if part_clean and part_clean not in seen:
            seen.add(part_clean)
            unique_parts.append(part_clean)
    
    # Combine with appropriate separators
    combined = []
    for i, part in enumerate(unique_parts):
        if i == 0:
            combined.append(part)
        elif part.startswith(':') or part.startswith('-'):
            # This is likely a continuation or list item
            combined.append(f" {part}")
        else:
            # This is likely a new sentence or section
            combined.append(f"; {part}")
    
    return "".join(combined)


def _create_chunks_from_headings_only(structured_content: Dict[str, Any], pdf_filename: str) -> List[Dict[str, Any]]:
    """Fallback method when PDF text extraction fails."""
    outline = structured_content.get('outline', [])
    title = structured_content.get('title', 'Untitled Document')
    
    chunks = []
    
    # Add title as first chunk if it exists and is meaningful
    if title and title.strip() and title != 'Untitled Document':
        chunks.append({
            'text_content': title.strip(),
            'source_document': pdf_filename,
            'page_number': 1,
            'heading_level': 'TITLE',
            'heading_text': title.strip()
        })
    
    # Process outline entries (headings only)
    for i, heading in enumerate(outline):
        level = heading.get('level', 'H1')
        text = heading.get('text', '').strip()
        page = heading.get('page', 1)
        
        if not text or _is_table_of_contents_entry(text):
            continue
        
        if level in ['H1', 'H2']:
            chunks.append({
                'text_content': text,
                'source_document': pdf_filename,
                'page_number': page,
                'heading_level': level,
                'heading_text': text
            })
    
    return chunks


def _extract_content_for_heading(
    heading_text: str, 
    heading_index: int, 
    outline: List[Dict], 
    all_blocks: List[TextBlock], 
    heading_to_block: Dict[str, TextBlock]
) -> str:
    """
    Extract all content that belongs to a specific heading using document order.
    
    Args:
        heading_text: The heading text to find content for
        heading_index: Index of this heading in the outline
        outline: Full outline from structured content
        all_blocks: All text blocks from the PDF (in document order)
        heading_to_block: Mapping of heading text to TextBlock objects
    
    Returns:
        Combined content including heading and all text until next same/higher level heading
    """
    content_parts = [heading_text]
    
    # Get current heading info
    current_heading = outline[heading_index]
    current_level = current_heading.get('level', 'H1')
    
    # Find the actual heading block in the document
    heading_block = _find_heading_block(heading_text, all_blocks)
    if not heading_block:
        safe_print(f"Warning: Could not find heading block for '{heading_text[:40]}...'")
        return heading_text
    
    # Find the document index of the heading block
    heading_block_index = None
    for i, block in enumerate(all_blocks):
        if block == heading_block:
            heading_block_index = i
            break
    
    if heading_block_index is None:
        safe_print(f"Warning: Could not find heading block index for '{heading_text[:40]}...'")
        return heading_text
    
    # Find the next heading of same or higher level
    next_heading_block_index = None
    for j in range(heading_index + 1, len(outline)):
        next_heading = outline[j]
        next_level = next_heading.get('level', 'H1')
        next_text = next_heading.get('text', '').strip()
        
        if _is_same_or_higher_level(next_level, current_level) and not _is_table_of_contents_entry(next_text):
            next_block = _find_heading_block(next_text, all_blocks)
            if next_block:
                for k, block in enumerate(all_blocks):
                    if block == next_block:
                        next_heading_block_index = k
                        break
                break
    
    # Determine the range of blocks to include
    start_index = heading_block_index + 1  # Start after the heading
    end_index = next_heading_block_index if next_heading_block_index else len(all_blocks)
    
    safe_print(f"Extracting content for '{heading_text[:40]}...' from block {start_index} to {end_index}")
    
    # Collect all text blocks between headings
    collected_blocks = []
    for i in range(start_index, end_index):
        block = all_blocks[i]
        
        # Skip table content (but be more lenient)
        if hasattr(block, 'is_in_table') and block.is_in_table:
            continue
            
        # Skip very short fragments (but allow mathematical symbols and Unicode)
        text = block.text.strip()
        if len(text) < 2:  # Only skip truly empty blocks
            continue
            
        # Skip blocks that look like other headings (but be more careful)
        if _is_likely_heading_block(block, all_blocks, i):
            continue
            
        collected_blocks.append(block)
    
    # Sort blocks by document order (they should already be sorted)
    collected_blocks.sort(key=lambda b: (b.page_number, getattr(b, 'y_position', 0)))
    
    # Add text content from collected blocks, preserving all characters including Unicode
    for block in collected_blocks:
        text = block.text.strip()
        if text:  # Include any non-empty text, including mathematical symbols
            content_parts.append(text)
    
    result = '\n\n'.join(content_parts)
    print(f"  -> Collected {len(collected_blocks)} text blocks, total length: {len(result)} chars")
    
    return result


def _find_heading_block(heading_text: str, all_blocks: List[TextBlock]) -> Optional[TextBlock]:
    """
    Find the TextBlock that corresponds to a heading text.
    Uses fuzzy matching to handle Unicode and formatting differences.
    """
    # First try exact match
    for block in all_blocks:
        if block.text.strip() == heading_text:
            return block
    
    # Try normalized matching (remove extra whitespace, normalize Unicode)
    import unicodedata
    
    def normalize_text(text: str) -> str:
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    normalized_heading = normalize_text(heading_text)
    
    for block in all_blocks:
        if normalize_text(block.text.strip()) == normalized_heading:
            return block
    
    # Try partial matching for truncated headings
    for block in all_blocks:
        block_text = normalize_text(block.text.strip())
        if (len(normalized_heading) > 20 and 
            (normalized_heading in block_text or block_text in normalized_heading)):
            # Check if it's a substantial match
            if len(set(normalized_heading.split()) & set(block_text.split())) >= 3:
                return block
    
    return None


def _is_likely_heading_block(block: TextBlock, all_blocks: List[TextBlock], block_index: int) -> bool:
    """
    Determine if a block is likely a heading that wasn't detected in the outline.
    More sophisticated than the previous version.
    """
    text = block.text.strip()
    
    # Don't skip very short text that might be mathematical expressions
    if len(text) < 5:
        return False
    
    # Don't skip text with lots of mathematical symbols or numbers
    import re
    if re.search(r'[∑∫∂∆∇≤≥≠±×÷√∞∈∉⊂⊃∪∩]', text):
        return False
    
    # Don't skip text that's mostly numbers/formulas
    if re.search(r'^\s*[\d\.\+\-\*\/\=\(\)\[\]]+\s*$', text):
        return False
    
    # Check formatting indicators
    formatting_score = 0
    
    if hasattr(block, 'is_bold') and block.is_bold:
        formatting_score += 2
    if hasattr(block, 'is_all_caps') and block.is_all_caps:
        formatting_score += 1
    if hasattr(block, 'is_underlined') and block.is_underlined:
        formatting_score += 1
    if hasattr(block, 'font_size') and block.font_size > 12:
        formatting_score += 1
    if hasattr(block, 'is_numbered') and block.is_numbered:
        formatting_score += 2
    
    # Check if it's isolated (has space around it)
    isolation_score = 0
    if block_index > 0:
        prev_block = all_blocks[block_index - 1]
        if hasattr(prev_block, 'space_after') and prev_block.space_after > 10:
            isolation_score += 1
    
    if block_index < len(all_blocks) - 1:
        next_block = all_blocks[block_index + 1]
        if hasattr(block, 'space_after') and block.space_after > 10:
            isolation_score += 1
    
    # Only consider it a heading if it has strong formatting AND isolation indicators
    return formatting_score >= 3 and isolation_score >= 1 and len(text) < 100


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity between two strings."""
    if not text1 or not text2:
        return 0.0
    
    # Simple similarity based on common words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def _looks_like_heading(block: TextBlock) -> bool:
    """Check if a text block looks like a heading that wasn't detected."""
    text = block.text.strip()
    
    # Very short text might be a heading
    if len(text) < 100 and (
        block.is_bold or 
        block.is_all_caps or 
        block.font_size > 12 or
        block.is_numbered
    ):
        return True
    
    return False


def _is_table_of_contents_entry(text: str) -> bool:
    """
    Check if text appears to be a table of contents entry.
    TOC entries often contain dots and page numbers.
    """
    # Look for patterns like "Chapter 1 .................. 5" or "1.1 Introduction .... 10"
    import re
    
    # Pattern for dots followed by numbers (page references)
    dots_pattern = r'\.{3,}\s*\d+\s*$'
    if re.search(dots_pattern, text):
        return True
    
    # Pattern for multiple sections listed together (like "2.1 ... 2.2 ... 2.3 ...")
    multiple_sections = text.count('2.') > 1 or text.count('3.') > 1 or text.count('4.') > 1
    if multiple_sections:
        return True
    
    # Very long lines with multiple page numbers are likely TOC
    page_numbers = re.findall(r'\b\d+\b', text)
    if len(page_numbers) > 3 and len(text) > 100:
        return True
    
    return False


def _is_same_or_higher_level(level1: str, level2: str) -> bool:
    """
    Check if level1 is same or higher priority than level2.
    H1 > H2 > H3 > H4, etc.
    """
    level_priority = {
        'TITLE': 0,
        'H1': 1,
        'H2': 2,
        'H3': 3,
        'H4': 4,
        'H5': 5,
        'H6': 6
    }
    
    priority1 = level_priority.get(level1, 999)
    priority2 = level_priority.get(level2, 999)
    
    return priority1 <= priority2


def load_structured_content(json_file_path: str) -> Dict[str, Any]:
    """
    Load structured content from JSON file created by heading_extractor.py.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the structured content
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file_path}: {e}")
        return {}


def save_chunks(chunks: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save chunks to a JSON file.
    
    Args:
        chunks: List of chunk dictionaries
        output_file: Path to output JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Chunks saved to: {output_file}")
    except Exception as e:
        print(f"Error saving chunks: {e}")


def main():
    """
    Example usage of the chunker module.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Create chunks from structured PDF content")
    parser.add_argument("json_file", help="JSON file from heading_extractor.py")
    parser.add_argument("pdf_filename", help="Original PDF filename")
    parser.add_argument("--output", "-o", help="Output JSON file for chunks")
    
    args = parser.parse_args()
    
    # Load structured content
    structured_content = load_structured_content(args.json_file)
    
    if not structured_content:
        print("No valid structured content found")
        return
    
    # Create chunks
    chunks = create_chunks(structured_content, args.pdf_filename)
    
    if not chunks:
        print("No chunks created")
        return
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        safe_print(f"{i}. {chunk['heading_level']}: {chunk['heading_text'][:60]}...")
        print(f"   Page: {chunk['page_number']}, Source: {chunk['source_document']}")
    
    # Save chunks if output file specified
    if args.output:
        save_chunks(chunks, args.output)
    else:
        # Print chunks as JSON
        print("\nChunks JSON:")
        print(json.dumps(chunks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()