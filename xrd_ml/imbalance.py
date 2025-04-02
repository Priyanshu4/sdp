""" Functions to handle imbalanced datasets.
"""

import numpy as np
from typing import Tuple, Optional

def resample_dataset_from_binned_solid_fractions(
        data: np.ndarray,
        solid_fractions: np.ndarray,
        n_bins: int = 20,
        n_samples_per_bin: Optional[int] = None,
        bin_undersampling_threshold: Optional[float] = None,
        oversample: bool = True,
        random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample the dataset to balance based on solid fractions.

    The dataset is split into bins based on solid fractions and an equal number of samples is drawn from each bin.
    If the number of samples in a bin is less than `n_samples_per_bin`, it samples with replacement.
    If the number of samples is greater than `n_samples_per_bin`, it samples without replacement.
    
    Args:
        data (np.ndarray): The dataset to resample.
        solid_fractions (np.ndarray): The solid fractions corresponding to the dataset.
        n_bins (int): The number of bins to split the solid fractions into.
        n_samples_per_bin (int, optional): The number of samples to draw from each bin. 
            If None, bin_undersampling_threshold is used to determine the number of samples.
        bin_undersampling_threshold (float, optional): The threshold for undersampling. If 0.8, n_samples is the number of samples in the 0.8 largest bin.
            If None, n_samples_per_bin is used.
        oversample (bool): Whether to oversample from bins with fewer samples.
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The resampled dataset and solid fractions.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if n_samples_per_bin is not None and bin_undersampling_threshold is not None:
        raise ValueError("Specify either n_samples_per_bin or bin_undersampling_threshold, not both.")
    if n_samples_per_bin is None and bin_undersampling_threshold is None:
        raise ValueError("Specify either n_samples_per_bin or bin_undersampling_threshold.")
    if n_samples_per_bin is not None:
        n_samples_per_bin = int(n_samples_per_bin)
    if bin_undersampling_threshold is not None:
        # Calculate the number of samples in the bin corresponding to the undersampling threshold
        bin_counts = np.histogram(solid_fractions, bins=n_bins)[0]
        sorted_bin_counts = np.sort(bin_counts)
        n_samples_per_bin = int(sorted_bin_counts[int(len(sorted_bin_counts) * bin_undersampling_threshold)])

    if n_samples_per_bin <= 0:
        raise ValueError("n_samples_per_bin must be greater than 0.")

    # Create bins for solid fractions
    bins = np.linspace(solid_fractions.min(), solid_fractions.max(), n_bins + 1)
    bin_indices = np.digitize(solid_fractions, bins) - 1

    # Initialize lists to hold resampled data and solid fractions
    resampled_data = []
    resampled_solid_fractions = []

    # Resample from each bin
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        bin_data = data[bin_mask]
        bin_solid_fractions = solid_fractions[bin_mask]

        # If the bin is empty, skip it
        if len(bin_data) == 0:
            continue

        # Randomly sample from the bin
        if len(bin_data) < n_samples_per_bin:
            size = n_samples_per_bin if oversample else len(bin_data)
            sampled_indices = np.random.choice(len(bin_data), size=size, replace=True)
        else:
            sampled_indices = np.random.choice(len(bin_data), size=n_samples_per_bin, replace=False)

        resampled_data.append(bin_data[sampled_indices])
        resampled_solid_fractions.append(bin_solid_fractions[sampled_indices])

    # Concatenate all resampled data and solid fractions
    resampled_data = np.concatenate(resampled_data)
    resampled_solid_fractions = np.concatenate(resampled_solid_fractions)

    return resampled_data, resampled_solid_fractions



def resample_dataset_percentile_threshold(
    data, 
    solid_fractions, 
    n_bins=20, 
    percentile_threshold=0.8,
    oversample=False,
    oversample_factor=0.5,
    random_seed=42,
    verbose=True
):
    """
    Resample a dataset to balance the distribution of solid fractions using a percentile threshold.
    
    This resampling method:
    1. Bins the data by solid fraction into n_bins equally spaced bins
    2. Finds the bin size at the specified percentile threshold
    3. Undersamples all bins with more samples than this threshold
    4. Optionally oversamples bins with very few samples
    
    Parameters:
        data: Features array (X)
        solid_fractions: Target values (y)
        n_bins: Number of bins to divide the solid fraction range [0,1]
        percentile_threshold: Percentile threshold for bin size capping (0.8 = 80th percentile)
        oversample: Whether to oversample underrepresented bins
        oversample_factor: Factor for determining which bins to oversample (bins < threshold * oversample_factor)
        random_seed: Random seed for reproducibility
        verbose: Whether to print resampling details
        
    Returns:
        Tuple of (resampled_data, resampled_solid_fractions)
    """
    np.random.seed(random_seed)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(solid_fractions, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins)-2)  # Ensure indices are within valid range
    
    # Count samples in each bin
    bin_counts = np.bincount(bin_indices, minlength=len(bins)-1)
    
    if verbose:
        print("Original distribution of solid fractions across bins:")
        for i in range(len(bins)-1):
            if bin_counts[i] > 0:
                bin_start, bin_end = bins[i], bins[i+1]
                print(f"  Bin {i+1}/{n_bins} ({bin_start:.2f}-{bin_end:.2f}): {bin_counts[i]} samples")
    
    # Find bin counts at the specified percentile threshold
    non_empty_bins = bin_counts[bin_counts > 0]
    percentile_count = np.percentile(non_empty_bins, percentile_threshold * 100)
    
    if verbose:
        print(f"Bin size at {percentile_threshold*100:.0f}th percentile: {percentile_count:.0f}")
    
    # Perform resampling
    X_resampled = []
    y_resampled = []
    
    for i in range(len(bins)-1):
        bin_mask = (bin_indices == i)
        X_bin = data[bin_mask]
        y_bin = solid_fractions[bin_mask]
        
        if len(X_bin) > 0:
            if len(X_bin) > percentile_count:
                # Under-sample bins larger than the percentile threshold
                keep_count = int(percentile_count)
                indices = np.random.choice(len(X_bin), keep_count, replace=False)
                X_resampled.append(X_bin[indices])
                y_resampled.append(y_bin[indices])
                if verbose:
                    print(f"  Bin {i+1}: Under-sampled from {len(X_bin)} to {keep_count}")
            elif oversample and len(X_bin) < percentile_count * oversample_factor:
                # Oversample very small bins if requested
                # First, keep all original samples
                X_resampled.append(X_bin)
                y_resampled.append(y_bin)
                
                # Then add duplicates to reach target
                target_count = int(percentile_count * oversample_factor)
                additional_needed = target_count - len(X_bin)
                if additional_needed > 0:
                    indices = np.random.choice(len(X_bin), additional_needed, replace=True)
                    X_resampled.append(X_bin[indices])
                    y_resampled.append(y_bin[indices])
                    if verbose:
                        print(f"  Bin {i+1}: Oversampled from {len(X_bin)} to {target_count}")
            else:
                # Keep bins below the threshold
                X_resampled.append(X_bin)
                y_resampled.append(y_bin)
                if verbose:
                    print(f"  Bin {i+1}: Kept all {len(X_bin)} samples (below threshold)")
    
    # Combine resampled data
    X_resampled_array = np.vstack(X_resampled)
    y_resampled_array = np.concatenate(y_resampled)
    
    # Verify new distribution
    if verbose:
        bin_indices_resampled = np.digitize(y_resampled_array, bins) - 1
        bin_indices_resampled = np.clip(bin_indices_resampled, 0, len(bins)-2)
        bin_counts_resampled = np.bincount(bin_indices_resampled, minlength=len(bins)-1)
        
        print("\nResampled distribution:")
        for i in range(len(bins)-1):
            if bin_counts_resampled[i] > 0:
                bin_start, bin_end = bins[i], bins[i+1]
                print(f"  Bin {i+1}/{n_bins} ({bin_start:.2f}-{bin_end:.2f}): {bin_counts_resampled[i]} samples")
        
        print(f"Original dataset: {len(solid_fractions)} samples")
        print(f"Resampled dataset: {len(y_resampled_array)} samples")
    
    return X_resampled_array, y_resampled_array

