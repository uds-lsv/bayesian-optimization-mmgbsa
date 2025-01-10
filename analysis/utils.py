from math import floor
import pandas as pd
import sqlite3
from pathlib import Path
import numpy as np
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Draw


def load_data(db, k_ratio=0.01):
    con = sqlite3.connect(db)
    data = pd.read_sql(
        """
        SELECT * FROM query_result
        JOIN experiment ON query_result.experiment_id == experiment.id
    """,
        con,
    )

    # Better names for plotting etc
    data = data.rename(columns={"iteration": "Iteration"})
    data["Embedding Model"] = data["embedding_model"].map(
        {"fingerprint": "Morgan Fingerprint", "chemberta-mtr": "ChemBERTa-2", "molformer": "MolFormer"}
    )
    data["Surrogate"] = data["surrogate"].map({
        "linear-empirical": "Linear",  "rf": "Random Forest"
    })
    data = data.drop(columns=["embedding_model", "surrogate"])

    data["Scoring"] = data["sampler"].map({
        "expected-improvement": "Expected Improvement",  "greedy": "Prediction", "random": "Random"
    })
    
    data["Iteration"] += 1

    # In our experiments we always create one database per dataset
    assert len(data["data_path"].unique()) == 1,  "Database contains results from more than 1 dataset."

    root = Path(__file__).parent.parent
    data_source =  Path(Path(data["data_path"].unique().item()).name)
    source_fp = root / "data" / data_source

    assert source_fp.exists(), source_fp
    source = pd.read_csv(source_fp)
    source = source.sort_values(by="target")

    # Calculate top_k hitrate
    k = int(floor(len(source) * k_ratio))
    top_k = source[:k]
    data["is_top_k"] = data["molecule"].isin(top_k["smiles"])
    data["is_top_k_cum"] = data.sort_values(by=["experiment_id", "Iteration"]).groupby("experiment_id")["is_top_k"].cumsum()

    assert (data.groupby(["experiment_id"])["is_top_k_cum"].max() == data.groupby(["experiment_id"])["is_top_k"].sum()).all()

    return data, top_k




def detect_step_changes(x, y, window_size=50, threshold=2, min_slope=1e-6):
    """
    Heuristic to detect significant step changes in cumulative data.
    
    Parameters:
    -----------
    x : array-like
        X-axis values (e.g., number of molecules screened)
    y : array-like
        Cumulative Y-axis values
    window_size : int
        Size of the rolling window to compute slope changes
    threshold : float
        Number of standard deviations above mean slope to be considered significant
    min_slope : float
        Minimum slope value to use when calculating relative changes
        to avoid division by zero
    
    Returns:
    --------
    list of dict
        List of detected step changes with their locations and magnitudes
    """
    # Calculate point-to-point slopes
    slopes = np.diff(y) / np.diff(x)
    
    # Use rolling window to compute local statistics
    step_changes = []
    for i in range(window_size, len(slopes) - window_size):
        # Get windows before and after current point
        window_before = slopes[i-window_size:i]
        window_after = slopes[i:i+window_size]
        
        # Compute statistics
        mean_before = np.mean(window_before)
        std_before = np.std(window_before)
        mean_after = np.mean(window_after)
        
        # Detect significant changes
        if mean_after > mean_before + threshold * std_before:
            # Perform t-test to confirm statistical significance
            t_stat, p_value = stats.ttest_ind(window_before, window_after)
            
            if p_value < 0.05:  # Statistical significance threshold
                # Calculate relative change with protection against division by zero
                if abs(mean_before) < min_slope:
                    # If starting from near-zero, use absolute change instead
                    relative_change = mean_after * 100  # Treat as 100% of the new value
                else:
                    relative_change = (mean_after - mean_before) / mean_before * 100
                
                step_changes.append({
                    'location': x[i],
                    'magnitude': mean_after - mean_before,
                    'p_value': p_value,
                    'relative_change': relative_change,
                    'mean_before': mean_before,
                    'mean_after': mean_after
                })
    
    # Merge nearby step changes
    merged_changes = []
    if step_changes:
        current_change = step_changes[0]
        for next_change in step_changes[1:]:
            if next_change['location'] - current_change['location'] < window_size:
                # Merge by taking the larger magnitude
                if next_change['magnitude'] > current_change['magnitude']:
                    current_change = next_change
            else:
                merged_changes.append(current_change)
                current_change = next_change
        merged_changes.append(current_change)
    
    return merged_changes


def mark_significant_changes(data, ax, model, mag_threshold=0.0):


    subset = data[data["Embedding Model"] == model]

    color = None
    for line in ax.get_lines():
        label = line.get_label()
        if label == model:
            color = line.get_color()
            break

    if color is None:
        raise ValueError(f"Could not fine line corresponding to {model}.")

        
    step_changes = detect_step_changes(subset["Iteration"].values, subset["is_top_k_cum"].values, window_size=50, threshold=2)


    sig_change_iterations = []
    for change in step_changes:

        if change["magnitude"] > mag_threshold:
            ax.axvline(change["location"], color=color, linestyle="dashed", alpha=0.3)
            sig_change_iterations.append(change["location"])

    sig_mols = subset[subset["Iteration"].isin(sig_change_iterations)]
    mols = sig_mols["molecule"].map(Chem.MolFromSmiles)
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=7,
        legends=[f"Iteration: {i}" for i in sig_change_iterations],
    )