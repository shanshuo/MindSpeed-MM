# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import copy
from typing import List


def balanced_bin_packing(seqlen_list: List[int], max_capacity: int):
    """Balanced bin packing algorithm that ensures each bin doesn't exceed max_capacity
    while maintaining load balance across bins.
    
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        max_capacity (int):
            maximum capacity for each bin/partition
            
    Returns:
        partitions (List[List[int]]):
            list of partitions, each containing indices of items
    """
    if not seqlen_list:
        return []
    
    # Create list of (seqlen, original_index) and sort by seqlen descending
    indexed_seqlens = [(seqlen, i) for i, seqlen in enumerate(seqlen_list)]
    indexed_seqlens.sort(reverse=True)  # Largest first (Best Fit Decreasing)
    
    # Initialize bins with their current capacity usage
    bins = []  # Each bin: {'items': [(idx, seqlen), ...], 'capacity_used': int}
    
    for seqlen, original_idx in indexed_seqlens:
        if seqlen > max_capacity:
            raise ValueError(f"Item with seqlen {seqlen} exceeds max_capacity {max_capacity}")
        
        # Find the best bin that can accommodate this item
        best_bin_idx = -1
        best_remaining_capacity = max_capacity + 1  # Initialize to impossible value
        
        for bin_idx, bin_info in enumerate(bins):
            remaining_capacity = max_capacity - bin_info['capacity_used']
            # Check if item fits and this bin has less remaining capacity (Best Fit)
            if remaining_capacity >= seqlen and remaining_capacity < best_remaining_capacity:
                best_bin_idx = bin_idx
                best_remaining_capacity = remaining_capacity
        
        if best_bin_idx != -1:
            # Add to existing bin
            bins[best_bin_idx]['items'].append((original_idx, seqlen))
            bins[best_bin_idx]['capacity_used'] += seqlen
        else:
            # Create new bin
            bins.append({
                'items': [(original_idx, seqlen)], 
                'capacity_used': seqlen
            })
    
    # Post-processing: Try to balance the bins by moving items between them
    # This helps reduce the variance in bin loads
    _balance_bins(bins, max_capacity)
    
    # Convert to partition format (list of indices for each partition)
    partitions = []
    for bin_info in bins:
        partition = [idx for idx, _ in bin_info['items']]
        partitions.append(partition)
    
    return partitions


def _balance_bins(bins: List[dict], max_capacity: int):
    """Helper function to balance loads across bins by moving items between bins.
    
    Parameters:
        bins: List of bin dictionaries with 'items' and 'capacity_used' keys
        max_capacity: Maximum capacity per bin
    """
    if len(bins) <= 1:
        return
    
    # Perform multiple passes to improve balance
    max_iterations = 3
    for _ in range(max_iterations):
        improved = False
        
        # Sort bins by current load
        bins.sort(key=lambda b: b['capacity_used'])
        
        # Try to move items from heaviest bins to lightest bins
        for heavy_idx in range(len(bins) - 1, 0, -1):
            heavy_bin = bins[heavy_idx]
            
            for light_idx in range(heavy_idx):
                light_bin = bins[light_idx]
                
                # Calculate load difference
                load_diff = heavy_bin['capacity_used'] - light_bin['capacity_used']
                if load_diff <= 1:  # Already balanced enough
                    break
                
                # Find items in heavy bin that can be moved to light bin
                for item_idx, (idx, seqlen) in enumerate(heavy_bin['items']):
                    new_light_load = light_bin['capacity_used'] + seqlen
                    new_heavy_load = heavy_bin['capacity_used'] - seqlen
                    
                    # Check if move is beneficial and doesn't violate capacity
                    if (new_light_load <= max_capacity and
                        abs(new_heavy_load - new_light_load) < load_diff):
                        
                        # Move the item
                        item = heavy_bin['items'].pop(item_idx)
                        light_bin['items'].append(item)
                        heavy_bin['capacity_used'] -= seqlen
                        light_bin['capacity_used'] += seqlen
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break


def rearrange_micro_batches(seqlen_list: List[int], max_token_len: int, dp_group=None):
    """Get order of seq lengths to make partitions balanced while ensuring 
    each partition doesn't exceed max_token_len capacity.
    
    This function uses a balanced bin packing algorithm that:
    1. Ensures each partition's total length <= max_token_len
    2. Minimizes the number of partitions needed
    3. Balances load across partitions to reduce variance
    
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        max_token_len (int):
            maximum token length per partition (capacity constraint)
        dp_group:
            distributed processing group for coordination
            
    Returns:
        partitions (List[List[int]]):
            list of partitions, each containing indices of items
    """
    if max(seqlen_list) > max_token_len:
        raise ValueError(f"seqlen of items:[{max(seqlen_list)}] must <= max_token_len:[{max_token_len}]")
    
    # Use balanced bin packing algorithm with capacity constraints
    partitions = balanced_bin_packing(seqlen_list=seqlen_list, max_capacity=max_token_len)
    
    return partitions


def get_reverse_idx(idx_map):
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map