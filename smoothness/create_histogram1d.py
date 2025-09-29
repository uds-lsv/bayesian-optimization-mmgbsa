#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from scipy.stats import skew

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

def compute_ratio(diffs, distances):
    ratios = np.divide(diffs, distances, out=np.full_like(diffs, np.inf), where=(distances != 0))
    replacement_value = get_power_of_10_replacement(ratios)
    ratios[distances == 0] = replacement_value
    return ratios


parser = argparse.ArgumentParser(description='Compute activity difference/distance ratios from .npy files')
parser.add_argument('distance_file', type=Path, help='Distance vector .npy file (<model>_distance.npy)')
parser.add_argument('outdir', type=Path, help='Output directory for ratio file')

args = parser.parse_args()

# Extract names from filenames
model_name = args.distance_file.stem.replace('_distance', '')
activity_diff_file_mmgbsa = args.distance_file.parent / f"mmgbsa_distance.npy"
activity_diff_file_vina = args.distance_file.parent / f"vina_distance.npy"

# Load data from .npy files
print(f"Loading distance data from {args.distance_file}")
distances = np.load(args.distance_file)

mmgbsa_diffs = np.load(activity_diff_file_mmgbsa)
vina_diffs = np.load(activity_diff_file_vina)

# Validate input
assert len(distances) == len(mmgbsa_diffs), "Distance and activity difference vectors must have same length"

print("Creating plots...")

# Create figure and axes
fig, (axes) = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axes.flatten()

# ========== Plot 1: Raw differences ==========
print("Plotting raw differences...")
bins_diffs = np.linspace(
    min(mmgbsa_diffs.min(), vina_diffs.min()),
    max(mmgbsa_diffs.max(), vina_diffs.max()),
    10
)
ax1.hist(mmgbsa_diffs, bins=bins_diffs, label="MMGBSA")
ax1.hist(vina_diffs, bins=bins_diffs, label="Vina")
ax1.set_yscale("log")
ax1.legend()
ax1.set_ylabel("Count")
ax1.set_title("Abs. difference activity")

# ========== Plot 2: Ratios ==========
print("Plotting ratios...")
mmgbsa_ratios = compute_ratio(mmgbsa_diffs, distances)
vina_ratios = compute_ratio(vina_diffs, distances)
bins_ratios = np.linspace(
    min(mmgbsa_ratios.min(), vina_ratios.min()),
    max(mmgbsa_ratios.max(), vina_ratios.max()),
    10
)
ax2.set_title("Ratio")
ax2.hist(mmgbsa_ratios, bins=bins_ratios, label="MMGBSA")
ax2.hist(vina_ratios, bins=bins_ratios, label="Vina")
ax2.set_yscale("log")
ax2.legend()
del mmgbsa_ratios, vina_ratios  # Free memory

# ========== Plot 3: MinMax scaled ==========
print("Plotting scaled differences...")
mmgbsa_scaled = minmax_scale(mmgbsa_diffs.reshape(-1, 1))
vina_scaled = minmax_scale(vina_diffs.reshape(-1, 1))
bins_scaled = np.linspace(0, 1, 10)
ax3.set_title("MinMax scaled")
ax3.hist(mmgbsa_scaled, bins=bins_scaled, label="MMGBSA", alpha=0.3)
ax3.hist(vina_scaled, bins=bins_scaled, label="Vina", alpha=0.3)
ax3.set_yscale("log")
ax3.set_ylabel("Count")
ax3.legend()

# ========== Plot 4: Scaled ratios ==========
print("Plotting scaled ratios...")
mmgbsa_scaled_ratios = compute_ratio(mmgbsa_scaled, distances.reshape(-1, 1))
del mmgbsa_scaled # Free memory
vina_scaled_ratios = compute_ratio(vina_scaled, distances.reshape(-1, 1))
del vina_scaled  # Free memory
bins_scaled_ratios = np.linspace(
    min(mmgbsa_scaled_ratios.min(), vina_scaled_ratios.min()),
    max(mmgbsa_scaled_ratios.max(), vina_scaled_ratios.max()),
    10
)
ax4.set_title("MinMax scaled")
ax4.hist(mmgbsa_scaled_ratios, bins=bins_scaled_ratios, label="MMGBSA", alpha=0.3)
ax4.hist(vina_scaled_ratios, bins=bins_scaled_ratios, label="Vina", alpha=0.3)
ax4.set_yscale("log")
ax4.legend()
del mmgbsa_scaled_ratios, vina_scaled_ratios  # Free memory

fig.tight_layout()
fig.savefig(args.outdir / f"{model_name}.png")
print(f"Saved plot to {args.outdir / f'{model_name}.png'}")