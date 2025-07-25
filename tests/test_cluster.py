"""
Unit tests for the font size clustering module.
"""

import pytest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cluster import collect_font_sizes, cluster_font_sizes, assign_size_levels, process_font_clustering


def test_collect_font_sizes():
    """Test font size collection from blocks."""
    blocks = [
        {'size': 16.0, 'text': 'Title'},
        {'size': 14.0, 'text': 'Heading'},
        {'size': 12.0, 'text': 'Body'},
        {'size': 16.0, 'text': 'Another Title'},  # Duplicate size
        {'size': 10.0, 'text': 'Small text'},
    ]
    
    sizes = collect_font_sizes(blocks)
    
    # Should return unique sizes in descending order
    assert sizes == [16.0, 14.0, 12.0, 10.0]


def test_cluster_font_sizes_edge_case_single_size():
    """Test clustering with only one unique font size."""
    blocks = [
        {'size': 12.0, 'text': 'Text 1'},
        {'size': 12.0, 'text': 'Text 2'},
    ]
    
    size_to_level = cluster_font_sizes(blocks)
    
    # Should assign level 1 to all blocks
    assert size_to_level == {12.0: 1}


def test_cluster_font_sizes_normal_case():
    """Test clustering with multiple font sizes."""
    blocks = [
        {'size': 18.0, 'text': 'Title'},
        {'size': 16.0, 'text': 'Heading 1'},
        {'size': 14.0, 'text': 'Heading 2'},
        {'size': 12.0, 'text': 'Body'},
        {'size': 10.0, 'text': 'Small'},
    ]
    
    size_to_level = cluster_font_sizes(blocks)
    
    # Should have 5 levels (or fewer if clustering groups some)
    assert len(size_to_level) == 5
    assert all(1 <= level <= 5 for level in size_to_level.values())
    
    # Largest font should get level 1
    max_size = max(size_to_level.keys())
    assert size_to_level[max_size] == 1


def test_assign_size_levels():
    """Test size level assignment to blocks."""
    blocks = [
        {'size': 16.0, 'text': 'Title'},
        {'size': 12.0, 'text': 'Body'},
    ]
    
    size_to_level = {16.0: 1, 12.0: 3}
    
    updated_blocks = assign_size_levels(blocks, size_to_level)
    
    assert len(updated_blocks) == 2
    assert updated_blocks[0]['size_level'] == 1
    assert updated_blocks[1]['size_level'] == 3
    
    # Original blocks should not be modified
    assert 'size_level' not in blocks[0]
    assert 'size_level' not in blocks[1]


def test_assign_size_levels_missing_size():
    """Test handling of blocks without size information."""
    blocks = [
        {'text': 'No size info'},
        {'size': None, 'text': 'Null size'},
    ]
    
    size_to_level = {}
    
    updated_blocks = assign_size_levels(blocks, size_to_level)
    
    assert len(updated_blocks) == 2
    assert updated_blocks[0]['size_level'] == 5  # Default level
    assert updated_blocks[1]['size_level'] == 5  # Default level


def test_process_font_clustering_integration():
    """Test the complete font clustering pipeline."""
    blocks = [
        {'size': 18.0, 'text': 'Title'},
        {'size': 14.0, 'text': 'Heading'},
        {'size': 12.0, 'text': 'Body text'},
        {'size': 12.0, 'text': 'More body text'},
    ]
    
    result_blocks = process_font_clustering(blocks)
    
    assert len(result_blocks) == 4
    
    # All blocks should have size_level assigned
    for block in result_blocks:
        assert 'size_level' in block
        assert 1 <= block['size_level'] <= 5
    
    # Largest font should get level 1
    title_block = next(b for b in result_blocks if b['text'] == 'Title')
    assert title_block['size_level'] == 1


if __name__ == '__main__':
    pytest.main([__file__])