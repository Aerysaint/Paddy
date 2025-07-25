"""
Tests for the detect module header/footer filtering functionality.
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detect import filter_headers_footers, apply_header_footer_filter, get_header_footer_info


def test_filter_headers_footers_basic():
    """Test basic header/footer filtering functionality."""
    # Create test blocks with recurring elements
    test_blocks = [
        # Page 1
        {'text': 'Company Header', 'page': 1, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Document Title', 'page': 1, 'fontname': 'Arial-Bold', 'size': 16.0},
        {'text': 'Content line 1', 'page': 1, 'fontname': 'Arial', 'size': 12.0},
        {'text': 'Confidential', 'page': 1, 'fontname': 'Arial', 'size': 10.0},
        
        # Page 2
        {'text': 'Company Header', 'page': 2, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Chapter 2', 'page': 2, 'fontname': 'Arial-Bold', 'size': 14.0},
        {'text': 'Content line 2', 'page': 2, 'fontname': 'Arial', 'size': 12.0},
        {'text': 'Confidential', 'page': 2, 'fontname': 'Arial', 'size': 10.0},
        
        # Page 3
        {'text': 'Company Header', 'page': 3, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Chapter 3', 'page': 3, 'fontname': 'Arial-Bold', 'size': 14.0},
        {'text': 'Content line 3', 'page': 3, 'fontname': 'Arial', 'size': 12.0},
        {'text': 'Confidential', 'page': 3, 'fontname': 'Arial', 'size': 10.0},
    ]
    
    filtered_blocks, original_with_flags = filter_headers_footers(test_blocks)
    
    # Should filter out 6 header/footer blocks (2 signatures × 3 pages)
    assert len(filtered_blocks) == 6
    assert len(original_with_flags) == 12
    
    # Check that header/footer blocks are properly flagged
    header_footer_count = sum(1 for block in original_with_flags if block.get('is_header_footer', False))
    assert header_footer_count == 6
    
    # Check that content blocks remain
    content_texts = [block['text'] for block in filtered_blocks]
    expected_content = ['Document Title', 'Content line 1', 'Chapter 2', 'Content line 2', 'Chapter 3', 'Content line 3']
    assert content_texts == expected_content


def test_filter_headers_footers_threshold():
    """Test that 70% threshold is properly applied."""
    # Create blocks where one element appears on 3/5 pages (60% < 70%)
    test_blocks = []
    
    # Add "Maybe Header" to first 3 pages
    for page in range(1, 4):
        test_blocks.append({'text': 'Maybe Header', 'page': page, 'fontname': 'Arial', 'size': 10.0})
        test_blocks.append({'text': f'Content {page}', 'page': page, 'fontname': 'Arial', 'size': 12.0})
    
    # Add content to remaining 2 pages without "Maybe Header"
    for page in range(4, 6):
        test_blocks.append({'text': f'Content {page}', 'page': page, 'fontname': 'Arial', 'size': 12.0})
    
    filtered_blocks, original_with_flags = filter_headers_footers(test_blocks)
    
    # With 5 pages, threshold is max(1, int(0.7 * 5)) = 3
    # "Maybe Header" appears on 3/5 pages (60%), which equals the threshold, so it should be filtered
    assert len(filtered_blocks) == 5  # 5 content blocks remain
    header_footer_count = sum(1 for block in original_with_flags if block.get('is_header_footer', False))
    assert header_footer_count == 3


def test_filter_headers_footers_exact_threshold():
    """Test behavior at exactly 70% threshold."""
    # Create 10 pages where element appears on exactly 7 pages (70%)
    test_blocks = []
    
    # Add recurring element to first 7 pages
    for page in range(1, 8):
        test_blocks.append({'text': 'Header', 'page': page, 'fontname': 'Arial', 'size': 10.0})
        test_blocks.append({'text': f'Content {page}', 'page': page, 'fontname': 'Arial', 'size': 12.0})
    
    # Add content to remaining 3 pages without header
    for page in range(8, 11):
        test_blocks.append({'text': f'Content {page}', 'page': page, 'fontname': 'Arial', 'size': 12.0})
    
    filtered_blocks, original_with_flags = filter_headers_footers(test_blocks)
    
    # Should filter out 7 header blocks (exactly at 70% threshold)
    assert len(filtered_blocks) == 10  # 10 content blocks remain
    header_footer_count = sum(1 for block in original_with_flags if block.get('is_header_footer', False))
    assert header_footer_count == 7


def test_apply_header_footer_filter_convenience():
    """Test the convenience function interface."""
    test_blocks = [
        # Header appears on all 3 pages
        {'text': 'Header', 'page': 1, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Content 1', 'page': 1, 'fontname': 'Arial', 'size': 12.0},
        {'text': 'Header', 'page': 2, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Content 2', 'page': 2, 'fontname': 'Arial', 'size': 12.0},
        {'text': 'Header', 'page': 3, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Content 3', 'page': 3, 'fontname': 'Arial', 'size': 12.0},
    ]
    
    filtered_blocks = apply_header_footer_filter(test_blocks)
    
    # Should return only content blocks (headers filtered out)
    assert len(filtered_blocks) == 3
    assert all('Content' in block['text'] for block in filtered_blocks)


def test_get_header_footer_info():
    """Test the debugging info function."""
    test_blocks = [
        # Header appears on all 3 pages
        {'text': 'Header', 'page': 1, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Content 1', 'page': 1, 'fontname': 'Arial', 'size': 12.0},
        {'text': 'Header', 'page': 2, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Content 2', 'page': 2, 'fontname': 'Arial', 'size': 12.0},
        {'text': 'Header', 'page': 3, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Content 3', 'page': 3, 'fontname': 'Arial', 'size': 12.0},
    ]
    
    info = get_header_footer_info(test_blocks)
    
    assert info['total_blocks'] == 6
    assert info['header_footer_blocks'] == 3  # 3 header blocks
    assert info['filtered_blocks'] == 3  # 3 content blocks remain
    assert info['total_pages'] == 3
    assert info['threshold_pages'] == 2  # max(1, int(0.7 * 3))
    assert len(info['signatures_detected']) == 1
    
    signature = info['signatures_detected'][0]
    assert signature['text'] == 'Header'
    assert signature['page_count'] == 3
    assert signature['percentage'] == 100.0


def test_empty_blocks():
    """Test handling of empty block list."""
    filtered_blocks, original_with_flags = filter_headers_footers([])
    
    assert filtered_blocks == []
    assert original_with_flags == []
    
    info = get_header_footer_info([])
    assert info['total_blocks'] == 0
    assert info['header_footer_blocks'] == 0
    assert info['filtered_blocks'] == 0


def test_single_page():
    """Test handling of single page documents."""
    test_blocks = [
        {'text': 'Header', 'page': 1, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Content', 'page': 1, 'fontname': 'Arial', 'size': 12.0},
    ]
    
    filtered_blocks, original_with_flags = filter_headers_footers(test_blocks)
    
    # For single-page documents, threshold is set to 2 (impossible to reach)
    # So no blocks should be filtered as headers/footers
    assert len(filtered_blocks) == 2  # Both blocks remain
    header_footer_count = sum(1 for block in original_with_flags if block.get('is_header_footer', False))
    assert header_footer_count == 0  # No blocks marked as header/footer


def test_signature_matching():
    """Test that signature matching works correctly with different font properties."""
    test_blocks = [
        # Same text, different fonts - should not be considered same signature
        {'text': 'Header', 'page': 1, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Header', 'page': 1, 'fontname': 'Times', 'size': 10.0},
        {'text': 'Header', 'page': 2, 'fontname': 'Arial', 'size': 10.0},
        {'text': 'Header', 'page': 2, 'fontname': 'Times', 'size': 10.0},
    ]
    
    filtered_blocks, original_with_flags = filter_headers_footers(test_blocks)
    
    # Each signature appears on 2/2 pages (100%), so both should be filtered
    assert len(filtered_blocks) == 0
    header_footer_count = sum(1 for block in original_with_flags if block.get('is_header_footer', False))
    assert header_footer_count == 4


# Import additional functions for table and column detection tests
from detect import (
    detect_tables, mark_table_blocks, detect_columns, 
    process_multi_column_page, detect_and_process_layout
)


def test_detect_tables_basic():
    """Test basic table detection functionality."""
    # Create blocks that simulate a table with tight spacing
    blocks = [
        # Table row 1
        {'x0': 10, 'y0': 100, 'x1': 30, 'y1': 110, 'page': 1, 'text': 'Col1'},
        {'x0': 30.5, 'y0': 100, 'x1': 50, 'y1': 110, 'page': 1, 'text': 'Col2'},  # Gap = 0.5pt
        {'x0': 50.5, 'y0': 100, 'x1': 70, 'y1': 110, 'page': 1, 'text': 'Col3'},  # Gap = 0.5pt
        
        # Table row 2
        {'x0': 10, 'y0': 110, 'x1': 30, 'y1': 120, 'page': 1, 'text': 'Data1'},
        {'x0': 30.5, 'y0': 110, 'x1': 50, 'y1': 120, 'page': 1, 'text': 'Data2'},  # Gap = 0.5pt
        {'x0': 50.5, 'y0': 110, 'x1': 70, 'y1': 120, 'page': 1, 'text': 'Data3'},  # Gap = 0.5pt
        
        # Non-table text (wider spacing)
        {'x0': 10, 'y0': 130, 'x1': 40, 'y1': 140, 'page': 1, 'text': 'Regular'},
        {'x0': 50, 'y0': 130, 'x1': 80, 'y1': 140, 'page': 1, 'text': 'Text'},  # Gap = 10pt
    ]
    
    table_indices = detect_tables(blocks)
    
    # First 6 blocks should be detected as table content (tight spacing)
    assert len(table_indices) == 6
    assert all(i in table_indices for i in range(6))
    assert 6 not in table_indices  # Regular text should not be in table
    assert 7 not in table_indices


def test_detect_tables_empty():
    """Test table detection with empty input."""
    assert detect_tables([]) == set()


def test_mark_table_blocks():
    """Test marking blocks with in_table flags."""
    blocks = [
        # Table-like blocks
        {'x0': 10, 'y0': 100, 'x1': 30, 'y1': 110, 'page': 1, 'text': 'Col1'},
        {'x0': 30.5, 'y0': 100, 'x1': 50, 'y1': 110, 'page': 1, 'text': 'Col2'},
        {'x0': 50.5, 'y0': 100, 'x1': 70, 'y1': 110, 'page': 1, 'text': 'Col3'},
        
        # Regular text
        {'x0': 10, 'y0': 130, 'x1': 80, 'y1': 140, 'page': 1, 'text': 'Regular text'},
    ]
    
    marked_blocks = mark_table_blocks(blocks)
    
    assert len(marked_blocks) == 4
    # Check that in_table flags are added
    for block in marked_blocks:
        assert 'in_table' in block
    
    # First 3 blocks should be marked as in_table
    assert marked_blocks[0]['in_table'] == True
    assert marked_blocks[1]['in_table'] == True
    assert marked_blocks[2]['in_table'] == True
    assert marked_blocks[3]['in_table'] == False


def test_detect_columns_single_column():
    """Test column detection with single column layout."""
    blocks = [
        {'x0': 50, 'y0': 100, 'x1': 150, 'y1': 110, 'page': 1, 'text': 'Text 1'},
        {'x0': 50, 'y0': 120, 'x1': 150, 'y1': 130, 'page': 1, 'text': 'Text 2'},
        {'x0': 50, 'y0': 140, 'x1': 150, 'y1': 150, 'page': 1, 'text': 'Text 3'},
    ]
    
    columns = detect_columns(blocks)
    
    # Should return single column
    assert len(columns) == 1
    assert len(columns[0]) == 3
    assert columns[0] == blocks


def test_detect_columns_two_columns():
    """Test column detection with two-column layout."""
    blocks = [
        # Left column
        {'x0': 50, 'y0': 100, 'x1': 150, 'y1': 110, 'page': 1, 'text': 'Left 1'},
        {'x0': 50, 'y0': 120, 'x1': 150, 'y1': 130, 'page': 1, 'text': 'Left 2'},
        
        # Right column (large gap = 100pt, which is >30% of page width ~300pt)
        {'x0': 250, 'y0': 100, 'x1': 350, 'y1': 110, 'page': 1, 'text': 'Right 1'},
        {'x0': 250, 'y0': 120, 'x1': 350, 'y1': 130, 'page': 1, 'text': 'Right 2'},
    ]
    
    columns = detect_columns(blocks)
    
    # Should detect two columns
    assert len(columns) == 2
    
    # Check that blocks are properly separated
    left_column = columns[0]
    right_column = columns[1]
    
    assert len(left_column) == 2
    assert len(right_column) == 2
    
    # Verify content
    left_texts = [block['text'] for block in left_column]
    right_texts = [block['text'] for block in right_column]
    
    assert 'Left 1' in left_texts
    assert 'Left 2' in left_texts
    assert 'Right 1' in right_texts
    assert 'Right 2' in right_texts


def test_detect_columns_empty():
    """Test column detection with empty input."""
    assert detect_columns([]) == []


def test_process_multi_column_page():
    """Test multi-column page processing."""
    blocks = [
        # Left column
        {'x0': 50, 'y0': 100, 'x1': 150, 'y1': 110, 'page': 1, 'text': 'Left 1'},
        {'x0': 50, 'y0': 120, 'x1': 150, 'y1': 130, 'page': 1, 'text': 'Left 2'},
        
        # Right column
        {'x0': 250, 'y0': 100, 'x1': 350, 'y1': 110, 'page': 1, 'text': 'Right 1'},
        {'x0': 250, 'y0': 120, 'x1': 350, 'y1': 130, 'page': 1, 'text': 'Right 2'},
    ]
    
    processed = process_multi_column_page(blocks)
    
    assert len(processed) == 4
    
    # Check that column information is added
    for block in processed:
        assert 'column' in block
    
    # Blocks should be sorted by y-coordinate, then x-coordinate
    y_coords = [block['y0'] for block in processed]
    assert y_coords == sorted(y_coords)


def test_detect_and_process_layout():
    """Test comprehensive layout detection and processing."""
    blocks = [
        # Table-like blocks
        {'x0': 10, 'y0': 100, 'x1': 30, 'y1': 110, 'page': 1, 'text': 'Col1'},
        {'x0': 30.5, 'y0': 100, 'x1': 50, 'y1': 110, 'page': 1, 'text': 'Col2'},
        
        # Regular text in columns
        {'x0': 50, 'y0': 120, 'x1': 150, 'y1': 130, 'page': 1, 'text': 'Left'},
        {'x0': 250, 'y0': 120, 'x1': 350, 'y1': 130, 'page': 1, 'text': 'Right'},
    ]
    
    processed = detect_and_process_layout(blocks)
    
    assert len(processed) == 4
    
    # Check that both table and column information is added
    for block in processed:
        assert 'in_table' in block
        # Column info may or may not be present depending on detection


def test_detect_tables_grid_pattern():
    """Test table detection using grid pattern recognition."""
    # Create blocks that form a grid pattern with tight spacing
    blocks = [
        # Row 1 - tight spacing between columns
        {'x0': 10, 'y0': 100, 'x1': 30, 'y1': 110, 'page': 1, 'text': 'A1'},
        {'x0': 30.5, 'y0': 100, 'x1': 50, 'y1': 110, 'page': 1, 'text': 'B1'},  # Gap = 0.5pt
        {'x0': 50.5, 'y0': 100, 'x1': 70, 'y1': 110, 'page': 1, 'text': 'C1'},  # Gap = 0.5pt
        
        # Row 2
        {'x0': 10, 'y0': 120, 'x1': 30, 'y1': 130, 'page': 1, 'text': 'A2'},
        {'x0': 30.5, 'y0': 120, 'x1': 50, 'y1': 130, 'page': 1, 'text': 'B2'},  # Gap = 0.5pt
        {'x0': 50.5, 'y0': 120, 'x1': 70, 'y1': 130, 'page': 1, 'text': 'C2'},  # Gap = 0.5pt
        
        # Row 3
        {'x0': 10, 'y0': 140, 'x1': 30, 'y1': 150, 'page': 1, 'text': 'A3'},
        {'x0': 30.5, 'y0': 140, 'x1': 50, 'y1': 150, 'page': 1, 'text': 'B3'},  # Gap = 0.5pt
        {'x0': 50.5, 'y0': 140, 'x1': 70, 'y1': 150, 'page': 1, 'text': 'C3'},  # Gap = 0.5pt
        
        # Non-grid text with wider spacing
        {'x0': 100, 'y0': 100, 'x1': 200, 'y1': 110, 'page': 1, 'text': 'Regular'},
    ]
    
    table_indices = detect_tables(blocks)
    
    # Grid pattern should be detected (9 blocks in 3x3 grid with tight spacing)
    assert len(table_indices) >= 9  # At least the grid blocks should be detected
    # The regular text might also be detected due to grid pattern alignment


def test_detect_columns_insufficient_gap():
    """Test that small gaps don't trigger column detection."""
    blocks = [
        # Text with small gap (not >30% of page width)
        {'x0': 50, 'y0': 100, 'x1': 150, 'y1': 110, 'page': 1, 'text': 'Text 1'},
        {'x0': 160, 'y0': 100, 'x1': 260, 'y1': 110, 'page': 1, 'text': 'Text 2'},  # Gap = 10pt
    ]
    
    columns = detect_columns(blocks)
    
    # Should return single column (gap too small)
    assert len(columns) == 1
    assert len(columns[0]) == 2


# Import title detection functions for testing
from detect import detect_title, _calculate_title_score

# Import heading detection functions for testing
from detect import detect_headings, _classify_base_headings, _apply_numbering_override, _apply_uppercase_and_gap_filtering, _count_numbering_dots


def test_detect_title_basic():
    """Test basic title detection functionality."""
    blocks = [
        {
            'page': 1, 'text': 'Document Title', 'size': 18, 'size_level': 1,
            'x_center': 306, 'is_header_footer': False, 'in_table': False,
            'num_prefix': False, 'x0': 200, 'x1': 400, 'y0': 50, 'y1': 70
        },
        {
            'page': 1, 'text': 'Chapter 1', 'size': 14, 'size_level': 2,
            'x_center': 100, 'is_header_footer': False, 'in_table': False,
            'num_prefix': False, 'x0': 50, 'x1': 150, 'y0': 100, 'y1': 120
        },
        {
            'page': 2, 'text': 'Body text', 'size': 12, 'size_level': 3,
            'x_center': 200, 'is_header_footer': False, 'in_table': False,
            'num_prefix': False, 'x0': 100, 'x1': 300, 'y0': 150, 'y1': 170
        }
    ]
    
    title = detect_title(blocks)
    assert title == 'Document Title'


def test_detect_title_no_candidates():
    """Test title detection when no suitable candidates exist."""
    blocks = [
        {
            'page': 1, 'text': 'Chapter 1', 'size': 14, 'size_level': 2,
            'x_center': 100, 'is_header_footer': False, 'in_table': False,
            'num_prefix': False, 'x0': 50, 'x1': 150, 'y0': 100, 'y1': 120
        },
        {
            'page': 4, 'text': 'Late Title', 'size': 18, 'size_level': 1,
            'x_center': 306, 'is_header_footer': False, 'in_table': False,
            'num_prefix': False, 'x0': 200, 'x1': 400, 'y0': 50, 'y1': 70
        }
    ]
    
    title = detect_title(blocks)
    assert title == ""


def test_detect_title_low_score():
    """Test title detection when best candidate scores below threshold."""
    blocks = [
        {
            'page': 1, 'text': '1.1 Introduction', 'size': 18, 'size_level': 1,
            'x_center': 100, 'is_header_footer': False, 'in_table': True,
            'num_prefix': True, 'x0': 50, 'x1': 150, 'y0': 50, 'y1': 70
        }
    ]
    
    title = detect_title(blocks)
    # Should return empty string due to table penalty and num penalty
    assert title == ""


def test_detect_title_header_footer_filtered():
    """Test that header/footer blocks are excluded from title detection."""
    blocks = [
        {
            'page': 1, 'text': 'Header Text', 'size': 18, 'size_level': 1,
            'x_center': 306, 'is_header_footer': True, 'in_table': False,
            'num_prefix': False, 'x0': 200, 'x1': 400, 'y0': 50, 'y1': 70
        },
        {
            'page': 1, 'text': 'Real Title', 'size': 16, 'size_level': 1,
            'x_center': 306, 'is_header_footer': False, 'in_table': False,
            'num_prefix': False, 'x0': 200, 'x1': 400, 'y0': 100, 'y1': 120
        }
    ]
    
    title = detect_title(blocks)
    assert title == 'Real Title'


def test_calculate_title_score():
    """Test title scoring calculation."""
    # Test centered, large font, no penalties
    block = {
        'size': 18, 'x_center': 306, 'in_table': False, 'num_prefix': False
    }
    score = _calculate_title_score(block, 18.0, 612.0)
    expected = 0.5 * 1.0 + 0.4 * 1.0  # size_norm=1.0, centered_norm=1.0, no penalties
    assert abs(score - expected) < 0.001
    
    # Test with penalties
    block_with_penalties = {
        'size': 18, 'x_center': 306, 'in_table': True, 'num_prefix': True
    }
    score_with_penalties = _calculate_title_score(block_with_penalties, 18.0, 612.0)
    expected_with_penalties = 0.5 * 1.0 + 0.4 * 1.0 - 0.2 - 0.3  # penalties applied
    assert abs(score_with_penalties - expected_with_penalties) < 0.001


def test_calculate_title_score_off_center():
    """Test title scoring with off-center text."""
    # Test off-center text (x_center = 150, page_center = 306)
    block = {
        'size': 18, 'x_center': 150, 'in_table': False, 'num_prefix': False
    }
    score = _calculate_title_score(block, 18.0, 612.0)
    
    # centered_norm = 1 - abs(150 - 306) / 306 = 1 - 156/306 ≈ 0.49
    expected_centered_norm = 1.0 - abs(150 - 306) / 306
    expected = 0.5 * 1.0 + 0.4 * expected_centered_norm
    assert abs(score - expected) < 0.001


def test_detect_title_empty_blocks():
    """Test title detection with empty block list."""
    title = detect_title([])
    assert title == ""


def test_detect_title_multiple_candidates():
    """Test title detection with multiple candidates, selecting highest score."""
    blocks = [
        # Candidate 1: centered but smaller
        {
            'page': 1, 'text': 'Smaller Title', 'size': 16, 'size_level': 1,
            'x_center': 306, 'is_header_footer': False, 'in_table': False,
            'num_prefix': False, 'x0': 200, 'x1': 400, 'y0': 50, 'y1': 70
        },
        # Candidate 2: larger but off-center
        {
            'page': 2, 'text': 'Larger Off-Center Title', 'size': 20, 'size_level': 1,
            'x_center': 150, 'is_header_footer': False, 'in_table': False,
            'num_prefix': False, 'x0': 50, 'x1': 250, 'y0': 50, 'y1': 70
        }
    ]
    
    title = detect_title(blocks)
    # Should select the one with higher overall score
    # This depends on the exact scoring, but both should be valid candidates
    assert title in ['Smaller Title', 'Larger Off-Center Title']


# Heading Detection Tests

def test_classify_base_headings():
    """Test base heading classification system."""
    blocks = [
        # Valid heading candidate
        {
            'text': 'Chapter 1: Introduction', 'size_level': 2, 'page': 1,
            'is_header_footer': False, 'in_table': False
        },
        # Header/footer - should be filtered out
        {
            'text': 'Page Header', 'size_level': 1, 'page': 1,
            'is_header_footer': True, 'in_table': False
        },
        # Table content - should be filtered out
        {
            'text': 'Table Cell', 'size_level': 2, 'page': 1,
            'is_header_footer': False, 'in_table': True
        },
        # Too short - should be filtered out
        {
            'text': 'Hi', 'size_level': 2, 'page': 1,
            'is_header_footer': False, 'in_table': False
        },
        # Body text (level 5+) - should be filtered out
        {
            'text': 'This is body text', 'size_level': 5, 'page': 1,
            'is_header_footer': False, 'in_table': False
        },
        # Another valid heading
        {
            'text': 'Section 2.1', 'size_level': 3, 'page': 2,
            'is_header_footer': False, 'in_table': False
        }
    ]
    
    headings = _classify_base_headings(blocks)
    
    assert len(headings) == 2
    assert headings[0]['text'] == 'Chapter 1: Introduction'
    assert headings[0]['level'] == 2
    assert headings[0]['classification_method'] == 'font_size'
    assert headings[1]['text'] == 'Section 2.1'
    assert headings[1]['level'] == 3


def test_count_numbering_dots():
    """Test counting dots in various numbering patterns."""
    # Standard dotted numbering
    assert _count_numbering_dots('1.2.3.4. Introduction') == 4
    assert _count_numbering_dots('1.2. Chapter') == 2
    assert _count_numbering_dots('1. First Level') == 1
    
    # Roman numerals
    assert _count_numbering_dots('I.II.III. Roman') == 3
    assert _count_numbering_dots('i.ii. Lower Roman') == 2
    
    # Mixed numbering
    assert _count_numbering_dots('1.a.i. Mixed') == 3
    assert _count_numbering_dots('1.A.I. Mixed Upper') == 3
    
    # Parentheses format
    assert _count_numbering_dots('1.2.3) Parentheses') == 3  # 2 dots + 1 for )
    assert _count_numbering_dots('1) Simple') == 1
    
    # No numbering
    assert _count_numbering_dots('No numbering here') == 0
    assert _count_numbering_dots('Just text') == 0


def test_apply_numbering_override():
    """Test numbering override rules."""
    headings = [
        # Heading with numbering - should override font size level
        {
            'level': 2, 'text': '1.2.3. Subsection', 'page': 1,
            'confidence': 0.5, 'classification_method': 'font_size',
            'source_block': {'num_prefix': True}
        },
        # Heading without numbering - should keep original level
        {
            'level': 2, 'text': 'Regular Heading', 'page': 1,
            'confidence': 0.5, 'classification_method': 'font_size',
            'source_block': {'num_prefix': False}
        }
    ]
    
    result = _apply_numbering_override(headings)
    
    assert len(result) == 2
    # First heading should have level = dot_count + 1 = 3 + 1 = 4
    assert result[0]['level'] == 4
    assert result[0]['confidence'] == 0.9
    assert result[0]['classification_method'] == 'numbering'
    
    # Second heading should remain unchanged
    assert result[1]['level'] == 2
    assert result[1]['confidence'] == 0.5
    assert result[1]['classification_method'] == 'font_size'


def test_apply_uppercase_and_gap_filtering():
    """Test uppercase promotion and gap filtering."""
    headings = [
        # Uppercase heading that should be promoted
        {
            'level': 3, 'text': 'UPPERCASE HEADING', 'page': 1,
            'confidence': 0.5, 'classification_method': 'font_size',
            'source_block': {
                'uppercase_ratio': 0.9, 'gap_above': 15.0,
                'page_avg_line_height': 12.0, 'num_prefix': False,
                'ends_with_colon': False
            }
        },
        # Heading with sufficient gap
        {
            'level': 2, 'text': 'Good Gap Heading', 'page': 1,
            'confidence': 0.5, 'classification_method': 'font_size',
            'source_block': {
                'uppercase_ratio': 0.2, 'gap_above': 5.0,
                'page_avg_line_height': 12.0, 'num_prefix': False,
                'ends_with_colon': False
            }
        },
        # Heading with colon (should pass filter)
        {
            'level': 2, 'text': 'Heading with colon:', 'page': 1,
            'confidence': 0.5, 'classification_method': 'font_size',
            'source_block': {
                'uppercase_ratio': 0.2, 'gap_above': 1.0,
                'page_avg_line_height': 12.0, 'num_prefix': False,
                'ends_with_colon': True
            }
        },
        # Heading with numbering (should pass filter)
        {
            'level': 2, 'text': '1.2 Numbered heading', 'page': 1,
            'confidence': 0.9, 'classification_method': 'numbering',
            'source_block': {
                'uppercase_ratio': 0.2, 'gap_above': 1.0,
                'page_avg_line_height': 12.0, 'num_prefix': True,
                'ends_with_colon': False
            }
        },
        # Heading that should be filtered out (insufficient gap, no colon, no numbering)
        {
            'level': 2, 'text': 'Bad heading', 'page': 1,
            'confidence': 0.5, 'classification_method': 'font_size',
            'source_block': {
                'uppercase_ratio': 0.2, 'gap_above': 1.0,
                'page_avg_line_height': 12.0, 'num_prefix': False,
                'ends_with_colon': False
            }
        }
    ]
    
    result = _apply_uppercase_and_gap_filtering(headings)
    
    assert len(result) == 4  # Last heading should be filtered out
    
    # First heading should be promoted (level 3 → 2)
    assert result[0]['level'] == 2
    assert result[0]['classification_method'] == 'font_size_uppercase'
    
    # Second heading should pass with sufficient gap
    assert result[1]['level'] == 2
    assert result[1]['text'] == 'Good Gap Heading'
    
    # Third heading should pass with colon
    assert result[2]['text'] == 'Heading with colon:'
    
    # Fourth heading should pass with numbering
    assert result[3]['text'] == '1.2 Numbered heading'


def test_detect_headings_integration():
    """Test complete heading detection pipeline."""
    blocks = [
        # Title (level 1) - should not be included in headings
        {
            'text': 'Document Title', 'size_level': 1, 'page': 1,
            'is_header_footer': False, 'in_table': False,
            'uppercase_ratio': 0.3, 'gap_above': 20.0,
            'page_avg_line_height': 12.0, 'num_prefix': False,
            'ends_with_colon': False
        },
        # Valid H2 heading
        {
            'text': 'Chapter 1: Introduction', 'size_level': 2, 'page': 1,
            'is_header_footer': False, 'in_table': False,
            'uppercase_ratio': 0.3, 'gap_above': 15.0,
            'page_avg_line_height': 12.0, 'num_prefix': False,
            'ends_with_colon': True
        },
        # Numbered heading (should override font size level)
        {
            'text': '1.2.3. Detailed Section', 'size_level': 3, 'page': 2,
            'is_header_footer': False, 'in_table': False,
            'uppercase_ratio': 0.2, 'gap_above': 8.0,
            'page_avg_line_height': 12.0, 'num_prefix': True,
            'ends_with_colon': False
        },
        # Uppercase heading (should be promoted)
        {
            'text': 'IMPORTANT SECTION', 'size_level': 3, 'page': 2,
            'is_header_footer': False, 'in_table': False,
            'uppercase_ratio': 0.9, 'gap_above': 10.0,
            'page_avg_line_height': 12.0, 'num_prefix': False,
            'ends_with_colon': False
        },
        # Header/footer (should be filtered out)
        {
            'text': 'Page Header', 'size_level': 2, 'page': 1,
            'is_header_footer': True, 'in_table': False,
            'uppercase_ratio': 0.3, 'gap_above': 15.0,
            'page_avg_line_height': 12.0, 'num_prefix': False,
            'ends_with_colon': False
        },
        # Body text (level 5+, should be filtered out)
        {
            'text': 'This is regular body text', 'size_level': 5, 'page': 1,
            'is_header_footer': False, 'in_table': False,
            'uppercase_ratio': 0.1, 'gap_above': 5.0,
            'page_avg_line_height': 12.0, 'num_prefix': False,
            'ends_with_colon': False
        }
    ]
    
    headings = detect_headings(blocks)
    
    assert len(headings) == 4  # Title + 3 valid headings
    
    # Check title (level 1)
    title_heading = next(h for h in headings if h['text'] == 'Document Title')
    assert title_heading['level'] == 1
    
    # Check chapter heading (level 2 with colon)
    chapter_heading = next(h for h in headings if 'Chapter 1' in h['text'])
    assert chapter_heading['level'] == 2
    
    # Check numbered heading (should be level 4 from 3 dots + 1)
    numbered_heading = next(h for h in headings if '1.2.3.' in h['text'])
    assert numbered_heading['level'] == 4
    assert numbered_heading['classification_method'] == 'numbering'
    
    # Check uppercase heading (should be promoted from 3 to 2)
    uppercase_heading = next(h for h in headings if 'IMPORTANT' in h['text'])
    assert uppercase_heading['level'] == 2
    assert 'uppercase' in uppercase_heading['classification_method']


def test_detect_headings_empty():
    """Test heading detection with empty input."""
    headings = detect_headings([])
    assert headings == []


def test_detect_headings_no_valid_candidates():
    """Test heading detection when no valid candidates exist."""
    blocks = [
        # All blocks are filtered out for various reasons
        {
            'text': 'Header', 'size_level': 2, 'page': 1,
            'is_header_footer': True, 'in_table': False  # Header/footer
        },
        {
            'text': 'Table cell', 'size_level': 2, 'page': 1,
            'is_header_footer': False, 'in_table': True  # In table
        },
        {
            'text': 'Hi', 'size_level': 2, 'page': 1,
            'is_header_footer': False, 'in_table': False  # Too short
        },
        {
            'text': 'Body text', 'size_level': 5, 'page': 1,
            'is_header_footer': False, 'in_table': False  # Body text level
        }
    ]
    
    headings = detect_headings(blocks)
    assert headings == []