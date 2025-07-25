"""
Font size clustering module for PDF layout extraction.

This module implements K-means clustering to group font sizes into discrete levels,
enabling consistent heading classification across documents with varying font schemes.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def collect_font_sizes(blocks: List[Dict]) -> List[float]:
    """
    Collect all unique font sizes from filtered blocks with precision rounding.
    
    Rounds font sizes to 2 decimals before deduplication to avoid spurious splits
    from floating-point differences (e.g., 7.201800000000006 and 7.2017999999999915).
    
    Args:
        blocks: List of text blocks (after header/footer removal and column splitting)
        
    Returns:
        List of unique font sizes sorted in descending order (rounded to 2 decimals)
    """
    font_sizes = set()
    
    for block in blocks:
        if 'size' in block and block['size'] is not None:
            # Round to 2 decimals to avoid floating-point precision issues
            rounded_size = round(float(block['size']), 2)
            font_sizes.add(rounded_size)
    
    # Convert to sorted list (descending order for easier level assignment)
    unique_sizes = sorted(list(font_sizes), reverse=True)
    
    logger.info(f"Collected {len(unique_sizes)} unique font sizes (rounded): {unique_sizes}")
    return unique_sizes


def cluster_font_sizes(blocks: List[Dict]) -> Dict[float, int]:
    """
    Apply K-means clustering to font sizes and assign level numbers.
    
    Args:
        blocks: List of text blocks to cluster
        
    Returns:
        Dictionary mapping font size to cluster level (1=largest to 5=smallest)
    """
    # Collect unique font sizes
    unique_sizes = collect_font_sizes(blocks)
    unique_count = len(unique_sizes)
    
    # Handle edge case: if unique_count < 2, assign size_level = 1 to all blocks
    if unique_count < 2:
        logger.warning(f"Only {unique_count} unique font sizes found, assigning level 1 to all")
        size_to_level = {}
        for block in blocks:
            if 'size' in block and block['size'] is not None:
                size_to_level[float(block['size'])] = 1
        return size_to_level
    
    # Improved clustering: use adaptive K based on size distribution
    k = _determine_optimal_clusters(unique_sizes)
    logger.info(f"Applying K-means clustering with K={k} for {unique_count} unique sizes")
    
    try:
        # Prepare data for clustering (reshape for sklearn)
        sizes_array = np.array(unique_sizes).reshape(-1, 1)
        
        # Apply K-means clustering with improved parameters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
        cluster_labels = kmeans.fit_predict(sizes_array)
        centroids = kmeans.cluster_centers_.flatten()
        
        # Sort cluster centroids in descending order for level assignment
        # Create mapping from centroid value to level (1=largest, 2=next largest, etc.)
        sorted_centroid_indices = np.argsort(centroids)[::-1]  # Descending order
        centroid_to_level = {i: level + 1 for level, i in enumerate(sorted_centroid_indices)}
        
        # Map each font size to its cluster level
        size_to_level = {}
        for i, size in enumerate(unique_sizes):
            cluster_id = cluster_labels[i]
            level = centroid_to_level[cluster_id]
            size_to_level[size] = level
        
        logger.info(f"Font size clustering completed: {len(size_to_level)} mappings created")
        logger.debug(f"Size to level mapping: {size_to_level}")
        
        return size_to_level
        
    except Exception as e:
        logger.error(f"K-means clustering failed: {e}")
        # Fallback: assign levels based on raw font sizes
        return _fallback_size_assignment(unique_sizes)


def _determine_optimal_clusters(unique_sizes: List[float]) -> int:
    """
    Determine optimal number of clusters based on font size distribution.
    
    Improved algorithm for research papers that need proper H1/H2/H3 hierarchy.
    Caps at 5 levels maximum (H1-H5) to prevent excessive heading levels.
    
    Args:
        unique_sizes: List of unique font sizes in descending order
        
    Returns:
        Optimal number of clusters (K) - capped at 5 for heading levels H1-H5 only
    """
    unique_count = len(unique_sizes)
    
    if unique_count <= 2:
        return unique_count
    
    # For research papers, we need to identify distinct heading levels
    # Look for natural breaks in font size distribution
    
    # Calculate size differences and their relative magnitudes
    size_diffs = []
    for i in range(len(unique_sizes) - 1):
        diff = unique_sizes[i] - unique_sizes[i + 1]
        relative_diff = diff / unique_sizes[i] if unique_sizes[i] > 0 else 0
        size_diffs.append((diff, relative_diff))
    
    if not size_diffs:
        return min(4, unique_count)
    
    # Find significant gaps using both absolute and relative differences
    abs_diffs = [d[0] for d in size_diffs]
    rel_diffs = [d[1] for d in size_diffs]
    
    # Use median-based thresholds to be more robust
    abs_median = np.median(abs_diffs)
    abs_threshold = abs_median + 1.5 * np.std(abs_diffs)
    
    rel_median = np.median(rel_diffs)
    rel_threshold = rel_median + 1.0 * np.std(rel_diffs)
    
    # Count significant gaps (either large absolute or large relative difference)
    significant_gaps = 0
    for abs_diff, rel_diff in size_diffs:
        if abs_diff > abs_threshold or rel_diff > rel_threshold:
            significant_gaps += 1
    
    # For research papers, aim for 3-5 levels typically
    # Title (level 1), Main sections (level 2), Subsections (level 3), etc.
    if unique_count <= 3:
        optimal_k = unique_count
    elif significant_gaps == 0:
        optimal_k = 3  # Default: title, main sections, body text
    elif significant_gaps == 1:
        optimal_k = 4  # Title, main sections, subsections, body text
    elif significant_gaps == 2:
        optimal_k = 5  # Full hierarchy with sub-subsections
    else:
        # Calculate uncapped optimal based on gaps
        optimal_k = min(unique_count, significant_gaps + 2)
    
    # Cap at 5 levels maximum for heading levels H1-H5 only
    # Ensure H5+ levels are treated as body text rather than creating excessive heading levels
    max_levels = 5  # H1, H2, H3, H4, H5 maximum
    capped_k = min(optimal_k, max_levels)
    
    logger.debug(f"Font size analysis: {unique_count} sizes, {significant_gaps} significant gaps")
    logger.debug(f"Absolute threshold: {abs_threshold:.2f}, Relative threshold: {rel_threshold:.3f}")
    logger.info(f"Optimal K: {optimal_k}, Capped K: {capped_k} (max {max_levels} levels)")
    
    return capped_k


def _fallback_size_assignment(unique_sizes: List[float]) -> Dict[float, int]:
    """
    Fallback method to assign size levels based on raw font sizes.
    
    Args:
        unique_sizes: List of unique font sizes in descending order
        
    Returns:
        Dictionary mapping font size to level (1=largest to 5=smallest)
    """
    logger.warning("Using fallback size assignment based on raw font sizes")
    
    size_to_level = {}
    for i, size in enumerate(unique_sizes):
        # Assign levels 1-N based on size ranking (no cap)
        level = i + 1
        size_to_level[size] = level
    
    return size_to_level


def assign_size_levels(blocks: List[Dict], size_to_level: Dict[float, int]) -> List[Dict]:
    """
    Add size_level field to each block based on clustering results.
    
    Uses rounded font sizes (2 decimals) for consistent lookup in size_to_level mapping.
    
    Args:
        blocks: List of text blocks to update
        size_to_level: Dictionary mapping font size to cluster level
        
    Returns:
        Updated list of blocks with size_level field added
    """
    updated_blocks = []
    
    for block in blocks:
        # Create a copy of the block to avoid modifying the original
        updated_block = block.copy()
        
        if 'size' in block and block['size'] is not None:
            # Round font size to 2 decimals for consistent lookup
            font_size = round(float(block['size']), 2)
            
            # Assign size level from clustering results
            if font_size in size_to_level:
                updated_block['size_level'] = size_to_level[font_size]
            else:
                # Handle edge case where font size wasn't in clustering - assign conservative level
                fallback_level = max(size_to_level.values()) if size_to_level else 5
                logger.warning(f"Font size {font_size} not found in clustering results, assigning level {fallback_level}")
                updated_block['size_level'] = fallback_level
        else:
            # Handle blocks without size information - assign level 5 (body text level) as default
            default_level = max(size_to_level.values()) if size_to_level else 5
            logger.warning(f"Block without size information found, assigning level {default_level}")
            updated_block['size_level'] = default_level
        
        updated_blocks.append(updated_block)
    
    logger.info(f"Assigned size levels to {len(updated_blocks)} blocks")
    return updated_blocks


def process_font_clustering(blocks: List[Dict]) -> List[Dict]:
    """
    Complete font size clustering pipeline.
    
    Args:
        blocks: List of text blocks (after header/footer removal and column splitting)
        
    Returns:
        Updated blocks with size_level field added
    """
    logger.info("Starting font size clustering process")
    
    # Step 1: Cluster font sizes and get size-to-level mapping
    size_to_level = cluster_font_sizes(blocks)
    
    # Step 2: Assign size levels to all blocks
    updated_blocks = assign_size_levels(blocks, size_to_level)
    
    logger.info("Font size clustering process completed")
    return updated_blocks