#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path

def get_power_of_10_replacement(ratios):
    """Find largest power of 10 based on maximum finite ratio"""
    finite_ratios = ratios[np.isfinite(ratios)]
    
    if len(finite_ratios) == 0:
        # If no finite ratios, use a default large value
        return 1e6
    
    max_finite = np.max(finite_ratios)
    
    # Find the power of 10 that is >= max_finite
    log_max = np.log10(max_finite)
    
    # Ceiling to get the next power of 10
    ceiling_power = np.ceil(log_max)
    
    # Convert back to the actual power of 10
    replacement_value = 10 ** ceiling_power
    
    return replacement_value


parser = argparse.ArgumentParser(description='Compute activity difference/distance ratios from .npy files')
parser.add_argument('distance_file', type=Path, help='Distance vector .npy file (<model>_distance.npy)')
parser.add_argument('activity_diff_file', type=Path, help='Activity difference vector .npy file (<activity>_distance.npy)') 
parser.add_argument('outdir', type=Path, help='Output directory for ratio file')

args = parser.parse_args()

# Extract names from filenames
model_name = args.distance_file.stem.replace('_distance', '')
activity_name = args.activity_diff_file.stem.replace('_distance', '')

# Load data from .npy files
print(f"Loading distance data from {args.distance_file}")
distances = np.load(args.distance_file)

print(f"Loading activity difference data from {args.activity_diff_file}")
activity_diffs = np.load(args.activity_diff_file)

# Validate input
assert len(distances) == len(activity_diffs), "Distance and activity difference vectors must have same length"

# Handle division by zero using largest power of 10 method
zero_distance_count = np.sum(distances == 0)

# Calculate ratios
# Replace infinite with the largest power of 10
ratios = np.divide(activity_diffs, distances, out=np.full_like(activity_diffs, np.inf), where=(distances != 0))
replacement_value = get_power_of_10_replacement(ratios)
ratios[distances == 0] = replacement_value

print(f"Edges with zero distance: {zero_distance_count}")
print(f"Maximum non-zero distance: {np.max(distances[distances > 0]) if np.any(distances > 0) else 'N/A'}")
print(f"Replacement value (largest power of 10): {replacement_value}")
print(f"Replaced {zero_distance_count} zero distances with {replacement_value}")

# Calculate matrix size for info
# For n molecules, upper triangle has n*(n-1)/2 unique pairs
# Given n_edges, solve: n_edges = n*(n-1)/2 for n
# Rearranges to: n^2 - n - 2*n_edges = 0
# Quadratic formula gives: n = (1 + sqrt(1 + 8*n_edges)) / 2
n_edges = len(distances)
n_molecules = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges)) / 2))

# Save results
ratio_file_name = args.outdir / f"{model_name}_{activity_name}_sali.npy"
np.save(ratio_file_name, ratios)
print(f"Ratios saved to {ratio_file_name}")

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"Matrix size: {n_molecules}x{n_molecules}")
print(f"Total edges: {len(ratios)}")
print(f"Mean activity diff/distance ratio: {ratios.mean():.4f}")
print(f"Std activity diff/distance ratio: {ratios.std():.4f}")
print(f"Min activity diff/distance ratio: {ratios.min():.4f}")
print(f"Max activity diff/distance ratio: {ratios.max():.4f}")
print(f"Median activity diff/distance ratio: {np.median(ratios):.4f}")

# Show ratios that came from replaced zero distances
if zero_distance_count > 0:
    replaced_ratios = ratios[distances == 0]
    print(f"Ratios from replaced zero distances: {len(replaced_ratios)}")
    if len(replaced_ratios) > 0:
        print(f"  Mean: {replaced_ratios.mean():.4f}")
        print(f"  Max: {replaced_ratios.max():.4f}")
        print(f"  Min: {replaced_ratios.min():.4f}")