"""
Hierarchy assembly and JSON output generation module.

This module handles:
- Building valid heading hierarchy from detected headings
- Applying demotion rules for misaligned hierarchies
- Generating final JSON output structure
"""

import json
import logging
from typing import List, Dict, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)


def build_hierarchy(headings: List[Dict]) -> List[Dict]:
    """
    Assemble headings into valid hierarchical structure.
    
    This function implements hierarchy building by:
    1. Sorting detected headings by (page ASC, y0 ASC) for proper document order
    2. Creating outline list and processing headings sequentially
    3. Adding H1 headings directly to root outline
    4. For heading level N, finding most recent parent at level N-1
    5. Attaching heading as child under appropriate parent
    6. Maintaining valid tree structure throughout assembly process
    
    Args:
        headings: List of heading dictionaries with level, text, page, and source_block
        
    Returns:
        List of hierarchically organized heading dictionaries
    """
    if not headings:
        logger.info("No headings provided for hierarchy building")
        return []
    
    logger.info(f"Building hierarchy from {len(headings)} detected headings")
    
    # Sort headings by (page ASC, y0 ASC) for proper document order
    sorted_headings = _sort_headings_by_position(headings)
    logger.debug(f"Sorted {len(sorted_headings)} headings by document position")
    
    # Create outline list and process headings sequentially
    outline = []
    # Keep track of recent parents at each level for parent-child relationships
    level_stack = {}  # level -> most recent heading at that level
    
    for heading in sorted_headings:
        level = heading.get('level', 1)
        
        if level == 1:
            # Add H1 headings directly to root outline
            outline_entry = _create_outline_entry(heading)
            outline.append(outline_entry)
            
            # Update level stack - this H1 becomes the parent for future H2s
            level_stack[1] = outline_entry
            # Clear any deeper levels since we have a new H1
            level_stack = {k: v for k, v in level_stack.items() if k <= 1}
            
            logger.debug(f"Added H1 heading to root: '{heading['text'][:50]}...' on page {heading['page']}")
        else:
            # For heading level N, find most recent parent at level N-1
            parent_level = level - 1
            parent = level_stack.get(parent_level)
            
            outline_entry = _create_outline_entry(heading)
            
            if parent is not None:
                # Attach heading as child under appropriate parent
                if 'children' not in parent:
                    parent['children'] = []
                parent['children'].append(outline_entry)
                
                # Update level stack - this heading becomes the parent for future deeper levels
                level_stack[level] = outline_entry
                # Clear any deeper levels since we have a new heading at this level
                keys_to_remove = [k for k in level_stack.keys() if k > level]
                for k in keys_to_remove:
                    del level_stack[k]
                
                logger.debug(f"Added H{level} heading as child of H{parent_level}: '{heading['text'][:50]}...' on page {heading['page']}")
            else:
                # No valid parent found - apply demotion rules
                # The demotion function will handle level_stack updates
                _apply_demotion_rules(outline_entry, level, level_stack, outline)
                logger.debug(f"Applied demotion rules for H{level} heading: '{heading['text'][:50]}...' on page {heading['page']}")
                # Note: level_stack is updated inside _apply_demotion_rules
    
    logger.info(f"Hierarchy building complete: {len(outline)} entries in root outline")
    return outline


def _sort_headings_by_position(headings: List[Dict]) -> List[Dict]:
    """
    Sort detected headings by (page ASC, y0 ASC) for proper document order.
    
    Args:
        headings: List of heading dictionaries
        
    Returns:
        List of headings sorted by document position
    """
    def get_sort_key(heading: Dict) -> tuple:
        page = heading.get('page', 1)
        
        # Get y0 from source_block if available, otherwise use 0
        source_block = heading.get('source_block', {})
        y0 = source_block.get('y0', 0)
        
        return (page, y0)
    
    sorted_headings = sorted(headings, key=get_sort_key)
    
    # Log sorting details for debugging
    for i, heading in enumerate(sorted_headings[:5]):  # Log first 5 for brevity
        page = heading.get('page', 1)
        y0 = heading.get('source_block', {}).get('y0', 0)
        text = heading.get('text', '')[:30]
        logger.debug(f"Sorted heading {i+1}: page {page}, y0 {y0:.1f}, text '{text}...'")
    
    return sorted_headings


def _apply_demotion_rules(outline_entry: Dict, original_level: int, level_stack: Dict, outline: List[Dict]) -> None:
    """
    Apply demotion rules for misaligned hierarchies.
    
    When no parent exists at level N-1, demote heading to level N-1.
    Apply recursive demotion until valid parent is found or reach H1.
    Attach demoted heading under last available ancestor.
    Log demotion actions for debugging purposes.
    
    Args:
        outline_entry: The outline entry to place
        original_level: Original level of the heading
        level_stack: Dictionary mapping levels to most recent headings (modified in place)
        outline: Root outline list (modified in place)
    """
    current_level = original_level
    
    # Apply recursive demotion until valid parent is found or reach H1
    while current_level > 1:
        parent_level = current_level - 1
        parent = level_stack.get(parent_level)
        
        if parent is not None:
            # Found valid parent - attach as child
            if 'children' not in parent:
                parent['children'] = []
            
            # Update the entry's level to reflect demotion
            outline_entry['level'] = current_level
            parent['children'].append(outline_entry)
            
            # Update level stack - this demoted heading becomes the parent for future deeper levels
            level_stack[current_level] = outline_entry
            # Clear any deeper levels since we have a new heading at this level
            keys_to_remove = [k for k in level_stack.keys() if k > current_level]
            for k in keys_to_remove:
                del level_stack[k]
            
            logger.info(f"Demoted H{original_level} to H{current_level} and attached under H{parent_level}: '{outline_entry['text'][:50]}...'")
            return
        
        # No parent at this level either - demote further
        current_level -= 1
        logger.debug(f"No parent at level {parent_level}, demoting from H{current_level + 1} to H{current_level}")
    
    # Reached H1 level - add to root outline
    outline_entry['level'] = 1
    outline.append(outline_entry)
    
    # Update level stack - this demoted H1 becomes the parent for future H2s
    level_stack[1] = outline_entry
    # Clear any deeper levels since we have a new H1
    keys_to_remove = [k for k in level_stack.keys() if k > 1]
    for k in keys_to_remove:
        del level_stack[k]
    
    logger.info(f"Demoted H{original_level} to H1 and added to root: '{outline_entry['text'][:50]}...'")
    return


def _create_outline_entry(heading: Dict) -> Dict:
    """
    Create an outline entry from a heading dictionary.
    
    Args:
        heading: Heading dictionary with level, text, page
        
    Returns:
        Outline entry dictionary with level, text, page
    """
    level = heading.get('level', 1)
    text = heading.get('text', '')  # Preserve whitespace during creation
    page = heading.get('page', 1)
    
    return {
        'level': level,  # Keep as integer during hierarchy building
        'text': text,
        'page': page
    }


def flatten_outline(outline: List[Dict]) -> List[Dict]:
    """
    Flatten hierarchical outline into a flat list for JSON output.
    
    This function converts the nested hierarchy structure into a flat list
    while preserving the document order and heading levels.
    
    Args:
        outline: Hierarchical outline structure with potential children
        
    Returns:
        Flat list of outline entries with level, text, and page
    """
    flattened = []
    
    def _flatten_recursive(entries: List[Dict]) -> None:
        """Recursively flatten outline entries."""
        for entry in entries:
            # Add current entry to flattened list
            flat_entry = {
                'level': f"H{entry['level']}",  # Convert integer level to H1, H2, etc. format
                'text': entry['text'].strip(),  # Strip whitespace from heading text content
                'page': entry['page']  # Ensure 1-based page numbering in output
            }
            flattened.append(flat_entry)
            
            # Recursively process children if they exist
            if 'children' in entry and entry['children']:
                _flatten_recursive(entry['children'])
    
    _flatten_recursive(outline)
    return flattened


def format_json_structure(title: str, headings: List[Dict]) -> Dict[str, Any]:
    """
    Create JSON structure formatting for document outline.
    
    This function implements the core JSON formatting requirements:
    - Convert heading levels to "H1", "H2", "H3", etc. format
    - Strip whitespace from heading text content  
    - Ensure 1-based page numbering in output
    
    Args:
        title: Document title (can be empty string)
        headings: List of detected headings with level, text, page
        
    Returns:
        Dictionary with title and outline fields ready for JSON serialization
    """
    logger.info(f"Formatting JSON structure with title: '{title}' and {len(headings)} headings")
    
    # Build hierarchical structure first
    hierarchy = build_hierarchy(headings)
    
    # Flatten the hierarchy for JSON output
    flat_outline = flatten_outline(hierarchy)
    
    # Format each outline entry according to requirements
    formatted_outline = []
    for entry in flat_outline:
        formatted_entry = {
            'level': entry['level'],  # Already in "H1", "H2", etc. format
            'text': entry['text'].strip(),  # Strip whitespace from heading text content
            'page': entry['page']  # Ensure 1-based page numbering (already handled in detection)
        }
        formatted_outline.append(formatted_entry)
    
    # Create final JSON structure
    json_structure = {
        'title': title.strip() if title else "",  # Handle empty titles
        'outline': formatted_outline
    }
    
    logger.info(f"JSON structure formatted: title='{json_structure['title']}', outline_entries={len(formatted_outline)}")
    return json_structure


def validate_json_schema(json_data: Dict[str, Any]) -> bool:
    """
    Validate JSON structure matches required schema with title and outline fields.
    
    Required schema:
    {
        "title": "<string or empty>", 
        "outline": [
            {"level": "H1", "text": "...", "page": 1}, 
            {"level": "H2", "text": "...", "page": 2}, 
            ..., 
            {"level": "Hn", "text": "...", "page": p}
        ]
    }
    
    Args:
        json_data: Dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check top-level structure
        if not isinstance(json_data, dict):
            logger.error("JSON data is not a dictionary")
            return False
        
        # Check required fields exist
        if 'title' not in json_data:
            logger.error("Missing required 'title' field")
            return False
        
        if 'outline' not in json_data:
            logger.error("Missing required 'outline' field")
            return False
        
        # Validate title field
        title = json_data['title']
        if not isinstance(title, str):
            logger.error(f"Title must be string, got {type(title)}")
            return False
        
        # Validate outline field
        outline = json_data['outline']
        if not isinstance(outline, list):
            logger.error(f"Outline must be list, got {type(outline)}")
            return False
        
        # Validate each outline entry
        for i, entry in enumerate(outline):
            if not isinstance(entry, dict):
                logger.error(f"Outline entry {i} must be dictionary, got {type(entry)}")
                return False
            
            # Check required fields in outline entry
            required_fields = ['level', 'text', 'page']
            for field in required_fields:
                if field not in entry:
                    logger.error(f"Outline entry {i} missing required field '{field}'")
                    return False
            
            # Validate level field (must be H1, H2, H3, etc.)
            level = entry['level']
            if not isinstance(level, str) or not level.startswith('H'):
                logger.error(f"Outline entry {i} level must be string starting with 'H', got '{level}'")
                return False
            
            try:
                level_num = int(level[1:])
                if level_num < 1:
                    logger.error(f"Outline entry {i} level number must be >= 1, got {level_num}")
                    return False
            except ValueError:
                logger.error(f"Outline entry {i} level must be H followed by number, got '{level}'")
                return False
            
            # Validate text field
            text = entry['text']
            if not isinstance(text, str):
                logger.error(f"Outline entry {i} text must be string, got {type(text)}")
                return False
            
            # Validate page field (must be non-negative integer for 0-based page numbering)
            page = entry['page']
            if not isinstance(page, int) or page < 0:
                logger.error(f"Outline entry {i} page must be non-negative integer, got {page}")
                return False
        
        logger.debug(f"JSON schema validation passed: {len(outline)} outline entries")
        return True
        
    except Exception as e:
        logger.error(f"JSON schema validation failed with exception: {e}")
        return False


def handle_empty_cases(title: str, outline: List[Dict]) -> Dict[str, Any]:
    """
    Handle empty titles and empty outline arrays correctly.
    
    Args:
        title: Document title (may be None or empty)
        outline: List of outline entries (may be empty)
        
    Returns:
        Properly formatted JSON structure handling empty cases
    """
    # Handle empty title case
    if not title or not title.strip():
        formatted_title = ""
        logger.debug("Using empty string for missing/empty title")
    else:
        formatted_title = title.strip()
    
    # Handle empty outline case
    if not outline:
        formatted_outline = []
        logger.debug("Using empty array for missing/empty outline")
    else:
        formatted_outline = outline
    
    return {
        'title': formatted_title,
        'outline': formatted_outline
    }


def serialize_to_json(data: Dict[str, Any], indent: Optional[int] = 2) -> str:
    """
    Add JSON serialization with proper formatting.
    
    Args:
        data: Dictionary to serialize
        indent: Number of spaces for indentation (None for compact)
        
    Returns:
        JSON string with proper formatting
    """
    try:
        json_string = json.dumps(data, indent=indent, ensure_ascii=False, separators=(',', ': '))
        logger.debug(f"JSON serialization successful, length: {len(json_string)} characters")
        return json_string
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        raise


def generate_document_outline(title: str, headings: List[Dict]) -> Dict[str, Any]:
    """
    Generate complete document outline with validation and schema compliance.
    
    This function implements the complete output generation pipeline:
    - Format JSON structure with proper heading levels and text formatting
    - Validate schema compliance
    - Handle empty titles and outline arrays
    - Provide JSON serialization
    
    Args:
        title: Document title (can be empty string or None)
        headings: List of detected headings
        
    Returns:
        Validated JSON structure ready for output
        
    Raises:
        ValueError: If generated JSON fails schema validation
    """
    logger.info(f"Generating document outline for title: '{title}' with {len(headings)} headings")
    
    # Format JSON structure
    json_structure = format_json_structure(title or "", headings)
    
    # Handle empty cases
    json_structure = handle_empty_cases(json_structure['title'], json_structure['outline'])
    
    # Note: Using 1-based page numbering as per requirements (no conversion needed)
    
    # Validate schema compliance
    if not validate_json_schema(json_structure):
        error_msg = "Generated JSON structure failed schema validation"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Document outline generated successfully: title='{json_structure['title']}', entries={len(json_structure['outline'])}")
    return json_structure


def fix_page_numbering_for_single_page_docs(outline_entries: List[Dict], total_pages: int) -> List[Dict]:
    """
    Fix page numbering for single-page documents.
    Single-page documents should use page 0 in the output.
    Multi-page documents should use 1-based indexing.
    """
    if total_pages == 1:
        # For single-page documents, convert to 0-based indexing
        for entry in outline_entries:
            if entry.get('page', 1) == 1:
                entry['page'] = 0
    
    return outline_entries


def _get_total_pages_from_headings(headings: List[Dict]) -> int:
    """Get total number of pages from headings."""
    if not headings:
        return 1
    
    pages = set()
    for heading in headings:
        page = heading.get('page', 1)
        if isinstance(page, int) and page > 0:
            pages.add(page)
        
        # Also check source_block
        source_block = heading.get('source_block', {})
        if source_block and 'page' in source_block:
            source_page = source_block['page']
            if isinstance(source_page, int) and source_page > 0:
                pages.add(source_page)
    
    return max(pages) if pages else 1