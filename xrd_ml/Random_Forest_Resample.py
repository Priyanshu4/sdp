import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from plotting import plot_model_predictions_by_temp
import argparse

from train_test_split import (
    load_train_data, 
    load_validation_data, 
    load_test_data_by_temp,
    get_x_y_as_np_array,
    data_by_temp_to_x_y_np_array,
    TRAIN_TEST_SPLITS
)

def resample_dataset_from_binned_solid_fractions(data, solid_fractions, n_bins=20, 
                                                bin_undersampling_threshold=0.8, 
                                                oversample=False, random_seed=42):
    """
    Resample a dataset based on binned solid fractions.
    
    Parameters:
    - data: Input features (numpy array)
    - solid_fractions: Target values (numpy array)
    - n_bins: Number of bins to divide the solid fraction range into
    - bin_undersampling_threshold: Threshold for undersampling (fraction of median bin count)
    - oversample: Whether to oversample underrepresented bins
    - random_seed: Random seed for reproducibility
    
    Returns:
    - resampled_data: Resampled input features
    - resampled_solid_fractions: Resampled target values
    """
    np.random.seed(random_seed)
    
    # Create bins for solid fraction
    bins = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(solid_fractions, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins)-2)
    
    # Count samples in each bin
    bin_counts = np.bincount(bin_indices, minlength=len(bins)-1)
    print("Distribution of training data across solid fraction bins:")
    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_end = bins[i+1]
        print(f"  Bin {i} ({bin_start:.2f}-{bin_end:.2f}): {bin_counts[i]} samples")
    
    # Calculate target count (median of non-empty bins)
    non_empty_bins = bin_counts[bin_counts > 0]
    target_count = np.median(non_empty_bins)
    print(f"Target sample count per bin: {target_count}")
    
    # Prepare for resampling
    resampled_data = []
    resampled_solid_fractions = []
    
    # Process each bin
    for i in range(len(bins)-1):
        bin_mask = (bin_indices == i)
        bin_data = data[bin_mask]
        bin_sf = solid_fractions[bin_mask]
        
        if len(bin_data) == 0:
            continue  # Skip empty bins
        
        # Undersample bins with more than threshold * target_count samples
        if len(bin_data) > target_count * bin_undersampling_threshold:
            keep_count = int(target_count * bin_undersampling_threshold)
            indices = np.random.choice(len(bin_data), keep_count, replace=False)
            resampled_data.append(bin_data[indices])
            resampled_solid_fractions.append(bin_sf[indices])
        
        # Oversample bins with few samples (if enabled)
        elif oversample and len(bin_data) < target_count * (1 - bin_undersampling_threshold):
            # Keep all original samples
            resampled_data.append(bin_data)
            resampled_solid_fractions.append(bin_sf)
            
            # Add duplicates as needed
            additional_needed = int(target_count * 0.8) - len(bin_data)
            if additional_needed > 0:
                indices = np.random.choice(len(bin_data), additional_needed, replace=True)
                resampled_data.append(bin_data[indices])
                resampled_solid_fractions.append(bin_sf[indices])
        else:
            # Keep bins with moderate sample counts unchanged
            resampled_data.append(bin_data)
            resampled_solid_fractions.append(bin_sf)
    
    # Combine resampled data
    resampled_data = np.vstack(resampled_data)
    resampled_solid_fractions = np.concatenate(resampled_solid_fractions)
    
    # Verify distribution after resampling
    bin_indices_resampled = np.digitize(resampled_solid_fractions, bins) - 1
    bin_counts_resampled = np.bincount(bin_indices_resampled, minlength=len(bins)-1)
    print("Distribution after resampling:")
    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_end = bins[i+1]
        print(f"  Bin {i} ({bin_start:.2f}-{bin_end:.2f}): {bin_counts_resampled[i]} samples")
    
    return resampled_data, resampled_solid_fractions

class XRDRandomForest:
    def __init__(self):
        # Set default parameters for the Random Forest
        self.params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',  
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
        self.model = RandomForestRegressor(**self.params)
        
    def plot_predictions_by_temp(self, y_true, y_pred, temps, output_name='random_forest_test_predictions_vs_actual.png', title='Random Forest: Predictions vs Actual Values (Test Data)'):
        plt.figure(figsize=(8, 6))
    
        plot_model_predictions_by_temp(y_true, y_pred, temps)
    
        plt.title(title)
        plt.grid(True)
        plt.savefig(output_name)
        plt.close()

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Random Forest model on training data."""
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluate model performance using common regression metrics."""
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        return predictions, metrics

    def plot_feature_importance(self, feature_names=None, output_name='random_forest_feature_importance.png'):
        """Plot and save the Random Forest feature importance scores."""
        importance = self.model.feature_importances_
        if feature_names is None:
            # Create default feature names if none provided
            feature_names = [f"Feature {i}" for i in range(len(importance))]
            
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = importance[indices]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_importance)), sorted_importance, align='center')
        plt.xticks(range(len(sorted_importance)), sorted_features, rotation=45, ha='right')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(output_name)
        plt.close()

    def plot_predictions(self, y_true, y_pred, output_name='random_forest_predictions_vs_actual.png', title='Random Forest: Predictions vs Actual Values'):
        """Plot predicted vs actual solid fraction values."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        # Plot reference line (perfect prediction)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Solid Fraction')
        plt.ylabel('Predicted Solid Fraction')
        plt.title(title)
        plt.grid(True)
        plt.savefig(output_name)
        plt.close()

    def evaluate_by_range(self, X_test, y_test):
        """Evaluate predictions across different solid fraction ranges."""
        predictions = self.model.predict(X_test)
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        
        results = {}
        for start, end in ranges:
            mask = (y_test >= start) & (y_test < end)
            if np.sum(mask) > 0:
                range_mse = mean_squared_error(y_test[mask], predictions[mask])
                range_mae = mean_absolute_error(y_test[mask], predictions[mask])
                results[f"{start:.1f}-{end:.1f}"] = {
                    "mse": range_mse,
                    "mae": range_mae,
                    "n_samples": np.sum(mask)
                }
        return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Random Forest model for XRD analysis.")
    parser.add_argument("--resample", action="store_true", 
                        help="Perform resampling to balance the training data distribution")
    parser.add_argument("--n-bins", type=int, default=20,
                       help="Number of bins for resampling (default: 20)")
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Threshold for undersampling bins (default: 0.8)")
    parser.add_argument("--oversample", action="store_true",
                       help="Enable oversampling of underrepresented bins")
    parser.add_argument("--split", type=str, default="original", choices=TRAIN_TEST_SPLITS.keys(),
                       help="Training/validation/test split to use (default: original)")
    parser.add_argument("--mode", type=str, default="validation", choices=["validation", "test"],
                       help="Evaluation mode: 'validation' or 'test' (default: validation)")
    args = parser.parse_args()

    # Use specified split
    split = TRAIN_TEST_SPLITS[args.split]
    print(f"Using {args.split} split:")
    print(f"  Training data: {split.train_data}")
    print(f"  Validation data: {split.validation_data}")
    print(f"  Test data: {split.test_data}")

    if args.mode == "test":
        # TEST MODE: Train on combined train+val data, evaluate on test data
        print("Using full training data (including validation) and test dataset for final evaluation.")
        
        # Load data
        print("Loading training data (including validation)...")
        train_data = load_train_data(split=split, suppress_load_errors=True, include_validation_set=True)
        
        print("Loading test data...")
        test_data_by_temp = load_test_data_by_temp(split=split, suppress_load_errors=True)
        
        # Convert to numpy arrays
        X_train, y_train = get_x_y_as_np_array(train_data)
        X_test, y_test, test_temps = data_by_temp_to_x_y_np_array(test_data_by_temp)
        
        print(f"Training data: {len(X_train)} samples")
        print(f"Test data: {len(X_test)} samples")
        
        # Apply resampling if requested
        if args.resample:
            print("Resampling the training dataset to balance it...")
            X_train, y_train = resample_dataset_from_binned_solid_fractions(
                data=X_train,
                solid_fractions=y_train,
                n_bins=args.n_bins,
                bin_undersampling_threshold=args.threshold,
                oversample=args.oversample,
                random_seed=42
            )
            print(f"Number of training samples after balancing: {X_train.shape[0]}")
        
        # Create descriptive filenames based on the split and mode
        suffix = "_resampled" if args.resample else ""
        plot_title = f"Random Forest: {args.split} Split (Test Mode){suffix}"
        feature_importance_filename = f"rf_{args.split}_test{suffix}_feature_importance.png"
        predictions_filename = f"rf_{args.split}_test{suffix}_predictions.png"
        
        # Initialize and train the model
        rf_model = XRDRandomForest()
        rf_model.train(X_train, y_train)
        
        # Evaluate on test data
        predictions, metrics = rf_model.evaluate(X_test, y_test)
        
        print("\nTest Data Performance:")
        print(f"Mean Squared Error: {metrics['mse']:.6f}")
        print(f"Mean Absolute Error: {metrics['mae']:.6f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
        print(f"R² Score: {metrics['r2']:.6f}")
        
        # Plot feature importance
        rf_model.plot_feature_importance(output_name=feature_importance_filename)
        
        # Plot predictions
        rf_model.plot_predictions_by_temp(
            y_test, predictions, test_temps,
            output_name=predictions_filename,
            title=plot_title
        )
        
        # Evaluate by range
        range_results = rf_model.evaluate_by_range(X_test, y_test)
        print("\nPerformance by solid fraction range on test data:")
        for range_name, met in range_results.items():
            print(f"\nRange {range_name}:")
            print(f"  MSE: {met['mse']:.6f}")
            print(f"  MAE: {met['mae']:.6f}")
            print(f"  Number of samples: {met['n_samples']}")
    
    else:
        # VALIDATION MODE: Train on train data, evaluate on validation data
        print("Using train data and validation dataset for evaluation.")
        
        # Load data
        print("Loading training data...")
        train_data = load_train_data(split=split, suppress_load_errors=True)
        
        print("Loading validation data with temperature information...")
        validation_data_by_temp = load_validation_data_by_temp(split=split, suppress_load_errors=True)
        
        # Convert to numpy arrays
        X_train, y_train = get_x_y_as_np_array(train_data)
        X_val, y_val, val_temps = data_by_temp_to_x_y_np_array(validation_data_by_temp)
        
        print(f"Training data: {len(X_train)} samples")
        print(f"Validation data: {len(X_val)} samples")
        
        # Print temperature information
        unique_temps = np.unique(val_temps, axis=0)
        print(f"Found {len(unique_temps)} unique temperature combinations in validation data:")
        for temp in unique_temps:
            count = np.sum(np.all(val_temps == temp, axis=1))
            print(f"  {temp[0]} K, Melting Temp {temp[1]} K: {count} samples")
        
        # Apply resampling if requested
        if args.resample:
            print("Resampling the training dataset to balance it...")
            X_train, y_train = resample_dataset_from_binned_solid_fractions(
                data=X_train,
                solid_fractions=y_train,
                n_bins=args.n_bins,
                bin_undersampling_threshold=args.threshold,
                oversample=args.oversample,
                random_seed=42
            )
            print(f"Number of training samples after balancing: {X_train.shape[0]}")
        
        # Create descriptive filenames based on the split and mode
        suffix = "_resampled" if args.resample else ""
        plot_title = f"Random Forest: {args.split} Split (Validation Mode){suffix}"
        feature_importance_filename = f"rf_{args.split}_val{suffix}_feature_importance.png"
        predictions_filename = f"rf_{args.split}_val{suffix}_predictions.png"
        
        # Initialize and train the model
        rf_model = XRDRandomForest()
        rf_model.train(X_train, y_train)
        
        # Evaluate overall performance on the validation data
        predictions, metrics = rf_model.evaluate(X_val, y_val)
        
        print("\nValidation Performance:")
        print(f"Mean Squared Error: {metrics['mse']:.6f}")
        print(f"Mean Absolute Error: {metrics['mae']:.6f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
        print(f"R² Score: {metrics['r2']:.6f}")
        
        # Plot feature importance
        rf_model.plot_feature_importance(output_name=feature_importance_filename)
        
        # Plot predictions
        rf_model.plot_predictions_by_temp(
            y_val, predictions, val_temps,
            output_name=predictions_filename,
            title=plot_title
        )
        
        # Evaluate by range
        range_results = rf_model.evaluate_by_range(X_val, y_val)
        print("\nPerformance by solid fraction range:")
        for range_name, met in range_results.items():
            print(f"\nRange {range_name}:")
            print(f"  MSE: {met['mse']:.6f}")
            print(f"  MAE: {met['mae']:.6f}")
            print(f"  Number of samples: {met['n_samples']}")

if __name__ == "__main__":
    main()
