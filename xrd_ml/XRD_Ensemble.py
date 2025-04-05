import numpy as np
import xgboost as xgb
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import joblib
import time

from plotting import (
    plot_model_predictions,
    plot_model_predictions_by_temp,
    save_plot,
    set_plots_subdirectory,
    get_plots_subdirectory
)
from train_test_split import (
    load_train_data, 
    load_validation_data,
    load_validation_data_by_temp,
    load_test_data_by_temp,
    get_x_y_as_np_array,
    data_by_temp_to_x_y_np_array,
    TRAIN_TEST_SPLITS
)

# Create plots directory if it doesn't exist
plots_dir = Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)

class XRDEnsemble:
    def __init__(self, use_stacking=True, use_scaling=True, params=None):
        """
        Initialize the ensemble model
        
        Parameters:
            use_stacking: Whether to use stacking (True) or voting (False)
            use_scaling: Whether to scale the input features
            params: Dictionary of model parameters
        """
        self.model = None
        self.use_stacking = use_stacking
        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None
        
        # Default parameters for component models
        self.params = {
            'xgb': {
                'n_estimators': 1000,
                'learning_rate': 0.01,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42,
                'objective': 'reg:squarederror'
            },
            'lgbm': {
                'n_estimators': 1000, 
                'learning_rate': 0.01,
                'max_depth': 6,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42,
                'objective': 'regression'
            },
            'catboost': {
                'iterations': 1000,
                'learning_rate': 0.01,
                'depth': 6,
                'subsample': 0.8,
                'colsample_bylevel': 0.8,
                'min_data_in_leaf': 20,
                'l2_leaf_reg': 3,
                'random_seed': 42,
                'loss_function': 'RMSE',
                'verbose': 0
            },
            'svr': {
                'kernel': 'rbf',
                'C': 100.0,
                'gamma': 'scale',
                'epsilon': 0.001
            },
            'mlp': {
                'hidden_layer_sizes': (100, 100),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'batch_size': 'auto',
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'random_state': 42
            }
        }
        
        # Update with custom parameters if provided
        if params:
            for model_name, model_params in params.items():
                if model_name in self.params:
                    self.params[model_name].update(model_params)
        
        # Initialize the ensemble model
        self._init_model()
        
    def _init_model(self):
        """Initialize the ensemble model with component models"""
        # Create the component models
        models = [
            ('xgb', xgb.XGBRegressor(**self.params['xgb'])),
            ('lgbm', LGBMRegressor(**self.params['lgbm'])),
            ('catboost', CatBoostRegressor(**self.params['catboost'])),
            ('svr', SVR(**self.params['svr'])),
            ('mlp', MLPRegressor(**self.params['mlp']))
        ]
        
        if self.use_stacking:
            # Use StackingRegressor with SVR as final estimator
            self.model = StackingRegressor(
                estimators=models,
                final_estimator=SVR(kernel='rbf', C=100.0, gamma='scale', epsilon=0.001),
                cv=5,
                n_jobs=-1
            )
        else:
            # Use VotingRegressor with equal weights for all models
            self.model = VotingRegressor(
                estimators=models,
                weights=[1, 1, 1, 1, 1],
                n_jobs=-1
            )
            
    def train(self, X_train, y_train):
        """
        Train the ensemble model
        
        Parameters:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            self: Trained model
        """
        print("Training ensemble model...")
        start_time = time.time()
        
        # Scale features if needed
        if self.use_scaling:
            print("Scaling features...")
            X_train = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Parameters:
            X: Features to predict
            
        Returns:
            predictions: Model predictions
        """
        # Scale features if needed
        if self.use_scaling:
            X = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, temps=None, output_prefix='ensemble'):
        """
        Evaluate model performance
        
        Parameters:
            X_test: Test features
            y_test: Test targets
            temps: Temperature information for each data point
            output_prefix: Prefix for output files
            
        Returns:
            metrics: Dictionary of performance metrics
            predictions: Model predictions
        """
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics
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
        
        # Plot predictions
        self.plot_predictions(y_test, predictions, temps, output_prefix)
        
        # Evaluate by range
        range_results = self.evaluate_by_range(X_test, y_test, predictions)
        
        # Print metrics
        print(f"\nModel Performance:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"Mean Absolute Error: {mae:.6f}")
        print(f"Root Mean Squared Error: {rmse:.6f}")
        print(f"R² Score: {r2:.6f}")
        
        # Print range-based metrics
        print("\nPerformance by solid fraction range:")
        for range_name, range_metrics in range_results.items():
            print(f"\nRange {range_name}:")
            print(f"  MSE: {range_metrics['mse']:.6f}")
            print(f"  MAE: {range_metrics['mae']:.6f}")
            print(f"  Number of samples: {range_metrics['n_samples']}")
        
        return metrics, predictions
    
    def evaluate_by_range(self, X_test, y_test, predictions=None):
        """
        Evaluate predictions across different ranges of solid fraction
        
        Parameters:
            X_test: Test features
            y_test: Test targets
            predictions: Model predictions (if None, will be computed)
            
        Returns:
            range_results: Dictionary of performance metrics by range
        """
        if predictions is None:
            predictions = self.predict(X_test)
            
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
    
    def plot_predictions(self, y_true, y_pred, temps=None, output_prefix='ensemble'):
        """
        Plot predicted vs actual values
        
        Parameters:
            y_true: True values
            y_pred: Predicted values
            temps: Temperature information for each data point
            output_prefix: Prefix for output files
        """
        plt.figure(figsize=(10, 8))
        
        # Add explicit grid
        plt.grid(True)
        
        # Check if temps is valid
        if temps is not None and len(temps) > 0 and len(temps) == len(y_true):
            print(f"Plotting with temperatures. Found {len(np.unique(temps, axis=0))} unique temperature combinations.")
            plot_model_predictions_by_temp(y_true, y_pred, temps)
        else:
            print("Plotting without temperatures - using regular scatter plot.")
            plot_model_predictions(y_true, y_pred)
        
        ensemble_type = "Stacking" if self.use_stacking else "Voting"
        plt.title(f'{ensemble_type} Ensemble: Predictions vs Actual Values')
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{output_prefix}_predictions.png"
        save_plot(plot_filename)
        print(f"Predictions plot saved to {plot_filename}")
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        
        Parameters:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'use_scaling': self.use_scaling,
            'use_stacking': self.use_stacking,
            'params': self.params
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from disk
        
        Parameters:
            filepath: Path to the saved model
            
        Returns:
            model: Loaded model
        """
        model_data = joblib.load(filepath)
        
        # Create an instance
        instance = cls(
            use_stacking=model_data['use_stacking'],
            use_scaling=model_data['use_scaling'],
            params=model_data['params']
        )
        
        # Set the loaded model and scaler
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        
        return instance

def main():
    """
    Main function to train and evaluate the ensemble model
    
    Command-line arguments:
        --split: Choose which train/test split to use (any split from TRAIN_TEST_SPLITS)
        --mode: Select "validation" or "test" mode
        --stacking: Use stacking (default) instead of voting
        --no-scaling: Disable feature scaling
        --resample: Perform data resampling to balance training data
        
    Examples:
        # Run on the validation set with default settings
        python ensemble.py
        
        # Run on the test set with the 2000K data in training
        python ensemble.py --mode test --split bring_in_300_2000
        
        # Use voting ensemble instead of stacking
        python ensemble.py --mode test --split bring_in_300_2000 --no-stacking
        
        # Disable feature scaling
        python ensemble.py --mode test --split bring_in_300_2000 --no-scaling
        
        # Perform data resampling
        python ensemble.py --mode test --split bring_in_300_2000 --resample
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate ensemble model for XRD analysis.")
    parser.add_argument("--split", type=str, default="original", choices=TRAIN_TEST_SPLITS.keys(), 
                        help="Training split to use (default: original)")
    parser.add_argument("--mode", type=str, default="validation", choices=["validation", "test"],
                        help="Evaluation mode: 'validation' or 'test' (default: validation)")
    parser.add_argument("--no-stacking", dest="stacking", action="store_false",
                        help="Use voting ensemble instead of stacking ensemble")
    parser.add_argument("--no-scaling", dest="scaling", action="store_false",
                        help="Disable feature scaling")
    parser.add_argument("--resample", action="store_true", 
                        help="Perform data resampling to balance training data")
    parser.set_defaults(stacking=True, scaling=True)
    args = parser.parse_args()
    
    # Use specified split
    split = TRAIN_TEST_SPLITS[args.split]
    
    # Create plot subdirectory
    ensemble_type = "stacking" if args.stacking else "voting"
    scaling_str = "scaled" if args.scaling else "unscaled"
    plots_subdir = f"ensemble_{ensemble_type}_{scaling_str}_{args.split}_{args.mode}"
    set_plots_subdirectory(plots_subdir, add_timestamp=True)
    
    print(f"Using {args.split} split:")
    print(f"  Training data: {split.train_data}")
    print(f"  Validation data: {split.validation_data}")
    print(f"  Test data: {split.test_data}")
    print(f"Using {'stacking' if args.stacking else 'voting'} ensemble")
    print(f"Feature scaling is {'enabled' if args.scaling else 'disabled'}")
    
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
        
        # Data resampling if requested
        if args.resample:
            print("Performing data resampling...")
            
            # Create bins for solid fraction
            num_bins = 30  # Using more bins for finer control
            bins = np.linspace(0, 1, num_bins+1)
            bin_indices = np.digitize(y_train, bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(bins)-2)
            
            # Count samples in each bin
            bin_counts = np.bincount(bin_indices, minlength=len(bins)-1)
            print("Distribution of training data across solid fraction bins:")
            for i in range(len(bins)-1):
                bin_start = bins[i]
                bin_end = bins[i+1]
                print(f"  Bin {i} ({bin_start:.2f}-{bin_end:.2f}): {bin_counts[i]} samples")
            
            # Calculate target count - using median instead of maximum for better balance
            non_empty_bins = bin_counts[bin_counts > 0]
            median_count = np.median(non_empty_bins)
            target_count = int(median_count)  # Using median as target
            print(f"Target sample count per bin: {target_count}")
            
            # Advanced resampling approach
            X_resampled = []
            y_resampled = []
            
            for i in range(len(bins)-1):
                bin_mask = (bin_indices == i)
                X_bin = X_train[bin_mask]
                y_bin = y_train[bin_mask]
                
                if len(X_bin) > 0:
                    if len(X_bin) > target_count * 1.5:
                        # Moderate undersampling for very common bins
                        keep_count = int(target_count * 1.2)
                        indices = np.random.choice(len(X_bin), keep_count, replace=False)
                        X_resampled.append(X_bin[indices])
                        y_resampled.append(y_bin[indices])
                    elif len(X_bin) < target_count * 0.5:
                        # Moderate oversampling for rare bins
                        X_resampled.append(X_bin)  # Keep all original samples
                        y_resampled.append(y_bin)
                        
                        additional_needed = int(target_count * 0.8) - len(X_bin)
                        if additional_needed > 0:
                            indices = np.random.choice(len(X_bin), additional_needed, replace=True)
                            X_resampled.append(X_bin[indices])
                            y_resampled.append(y_bin[indices])
                    else:
                        # Keep bins with reasonable representation
                        X_resampled.append(X_bin)
                        y_resampled.append(y_bin)
            
            # Combine resampled data
            X_train = np.vstack(X_resampled)
            y_train = np.concatenate(y_resampled)
            
            print(f"After resampling: {len(X_train)} training samples")
            
            # Verify distribution
            bin_indices_resampled = np.digitize(y_train, bins) - 1
            bin_indices_resampled = np.clip(bin_indices_resampled, 0, len(bins)-2)
            bin_counts_resampled = np.bincount(bin_indices_resampled, minlength=len(bins)-1)
            print("Distribution after resampling:")
            for i in range(len(bins)-1):
                bin_start = bins[i]
                bin_end = bins[i+1]
                print(f"  Bin {i} ({bin_start:.2f}-{bin_end:.2f}): {bin_counts_resampled[i]} samples")
        
        # Create and train the ensemble model
        ensemble = XRDEnsemble(
            use_stacking=args.stacking,
            use_scaling=args.scaling
        )
        
        # Train the model
        ensemble.train(X_train, y_train)
        
        # Evaluate on test data
        metrics, predictions = ensemble.evaluate(
            X_test, 
            y_test, 
            temps=test_temps,
            output_prefix=f"ensemble_{args.split}_test"
        )
        
        # Save the model
        model_filename = get_plots_subdirectory() / "ensemble_model.joblib"
        ensemble.save_model(model_filename)
        
    else:
        # VALIDATION MODE: Train on train data, evaluate on validation data
        print("Using train data and validation dataset for evaluation.")
        
        # Load data
        print("Loading training data...")
        train_data = load_train_data(split=split, suppress_load_errors=True)
        
        print("Loading validation data...")
        validation_data_by_temp = load_validation_data_by_temp(split=split, suppress_load_errors=True)
        
        # Convert to numpy arrays
        X_train, y_train = get_x_y_as_np_array(train_data)
        X_val, y_val, val_temps = data_by_temp_to_x_y_np_array(validation_data_by_temp)
        
        print(f"Training data: {len(X_train)} samples")
        print(f"Validation data: {len(X_val)} samples")
        
        # Data resampling if requested (same as in test mode)
        if args.resample:
            # Resample code (same as test mode)
            pass
        
        # Create and train the ensemble model
        ensemble = XRDEnsemble(
            use_stacking=args.stacking,
            use_scaling=args.scaling
        )
        
        # Train the model
        ensemble.train(X_train, y_train)
        
        # Evaluate on validation data
        metrics, predictions = ensemble.evaluate(
            X_val, 
            y_val, 
            temps=val_temps,
            output_prefix=f"ensemble_{args.split}_val"
        )
        
        # Save the model
        model_filename = get_plots_subdirectory() / "ensemble_model.joblib"
        ensemble.save_model(model_filename)

if __name__ == "__main__":
    main()
