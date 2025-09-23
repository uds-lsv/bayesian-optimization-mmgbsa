#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_triangular_matrix(values):
    """Create upper triangular matrix from condensed vector - memory efficient"""
    n_edges = len(values) 
    n = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges)) / 2))
    
    # Create empty matrix
    # nan is more memory efficient than 0
    matrix = np.full((n, n), np.nan)
    
    # Fill upper triangle
    triu_indices = np.triu_indices(n, k=1)
    matrix[triu_indices] = values
    
    return matrix

def visualize_matrices(distances, activity_diffs, ratios, save_path):
    """Create visualization with triangular matrices and ratio histogram - memory efficient"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate matrix size once
    n_edges = len(distances)
    n = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges)) / 2))
    print(f"Creating triangular matrices ({n}x{n})...")
    
    # Distance matrix - triangular only
    print("Creating distance matrix visualization...")
    dist_matrix = create_triangular_matrix(distances)
    im1 = axes[0,0].imshow(dist_matrix, cmap='Blues', origin='lower', rasterized=True)
    axes[0,0].set_title('Distance Matrix')
    axes[0,0].set_xlabel('Molecule')
    axes[0,0].set_ylabel('Molecule')
    plt.colorbar(im1, ax=axes[0,0])
    del dist_matrix
    
    # Activity difference matrix - triangular only
    print("Creating activity difference matrix visualization...")
    activity_diff_matrix = create_triangular_matrix(activity_diffs)
    im2 = axes[0,1].imshow(activity_diff_matrix, cmap='Reds', origin='lower', rasterized=True)
    axes[0,1].set_title('Activity Difference Matrix')
    axes[0,1].set_xlabel('Molecule')
    axes[0,1].set_ylabel('Molecule')
    plt.colorbar(im2, ax=axes[0,1])
    del activity_diff_matrix
    
    # Ratio matrix - triangular only
    print("Creating ratio matrix visualization...")
    ratio_matrix = np.log10(create_triangular_matrix(ratios))
    im3 = axes[1,0].imshow(ratio_matrix, cmap='RdYlBu_r', origin='lower', rasterized=True)
    axes[1,0].set_title('Activity Diff/Distance Ratio Matrix')
    axes[1,0].set_xlabel('Molecule')
    axes[1,0].set_ylabel('Molecule')
    plt.colorbar(im3, ax=axes[1,0])
    del ratio_matrix
    
    # Ratio histogram with summary statistics
    print("Creating ratio histogram...")
    axes[1,1].hist(ratios, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1,1].set_title('Activity Difference/Distance Ratio Distribution')
    axes[1,1].set_xlabel('Activity Difference / Distance Ratio')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_yscale("log")
    axes[1,1].grid(True, alpha=0.3)
    
    # Add summary statistics text to histogram
    stats_text = f'Mean: {ratios.mean():.3f}\n'
    stats_text += f'Std: {ratios.std():.3f}\n'
    stats_text += f'Median: {np.median(ratios):.3f}\n'
    stats_text += f'Min: {ratios.min():.3f}\n'
    stats_text += f'Max: {ratios.max():.3f}\n'
    stats_text += f'N: {len(ratios)}'
    
    # Position text box in upper right of histogram
    axes[1,1].text(0.98, 0.98, stats_text, transform=axes[1,1].transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


parser = argparse.ArgumentParser(description='Visualize activity difference/distance ratios from .npy files')
parser.add_argument('distance_file', type=Path, help='Distance vector .npy file (<model>_distance.npy)')
parser.add_argument('activity_diff_file', type=Path, help='Activity difference vector .npy file (<activity>_distance.npy)') 
parser.add_argument('ratios_file', type=Path, help='Computed ratios .npy file (<model>_<activity>_sali.npy)')
parser.add_argument('outdir', type=Path, help='Output directory for visualization')

args = parser.parse_args()

# Extract names from filenames for output naming
model_name = args.distance_file.stem.replace('_distance', '')
activity_name = args.activity_diff_file.stem.replace('_distance', '')

# Create output filename
output_filename = args.outdir / f"{model_name}_{activity_name}_activity_ratio_analysis.png"

# Load data from .npy files
print(f"Loading distance data from {args.distance_file}")
distances = np.load(args.distance_file)

print(f"Loading activity difference data from {args.activity_diff_file}")
activity_diffs = np.load(args.activity_diff_file)

print(f"Loading ratios data from {args.ratios_file}")
ratios = np.load(args.ratios_file)

# Validate input sizes
assert len(distances) == len(activity_diffs) == len(ratios), \
    "Distance, activity difference, and ratio vectors must have same length"

# Print basic info
n_edges = len(distances)
n_molecules = int(np.ceil((1 + np.sqrt(1 + 8 * n_edges)) / 2))
print(f"Matrix size: {n_molecules}x{n_molecules}")
print(f"Total edges: {n_edges}")

# Create visualization
visualize_matrices(distances, activity_diffs, ratios, output_filename)
print(f"Visualization saved to {output_filename}")