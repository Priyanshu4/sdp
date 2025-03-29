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