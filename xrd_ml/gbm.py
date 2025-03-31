import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from plotting import (
    plot_model_predictions,
    plot_model_predictions_by_temp,
    save_plot
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
from plotting import plot_solid_fraction_distribution

# Create plots directory if it doesn't exist
plots_dir = Path(__file__).parent.parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

class XRDBoost:
    def __init__(self, params=None):
        self.model = None
        self.num_boost_rounds = 1000
        
        # Default parameters
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.01,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42
        }
        
        # Update with custom parameters if provided
        if params:
            self.params.update(params)

    def train(self, X_train, y_train, X_val, y_val):
        """Train the XGBoost model with early stopping"""
        
        # Convert data to DMatrix format for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Setup evaluation list
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        print("Training XGBoost model...")
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_rounds,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        dtest = xgb.DMatrix(X_test)
        predictions = self.model.predict(dtest)
        
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
        
        return predictions, metrics

    def plot_feature_importance(self, output_name='xgboost_feature_importance.png'):
        """Plot feature importance scores"""
        importance_type = 'weight'  # Can also use 'gain' or 'cover'
        
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(self.model, importance_type=importance_type)
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(plots_dir / output_name)
        plt.close()
        print(f"Feature importance plot saved to {plots_dir / output_name}")

    def plot_predictions(self, y_true, y_pred, temps=None, output_name='xgboost_predictions.png', title=None):
        """
        Plot predicted vs actual values with temperature information
        
        Parameters:
            y_true: True values
            y_pred: Predicted values
            temps: Temperature information for each data point
            output_name: Name of the output file
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        
        # Add explicit grid
        plt.grid(True)
        
        # Check if temps is valid
        if temps is not None and len(temps) > 0 and len(temps) == len(y_true):
            print(f"Plotting with temperatures. Found {len(np.unique(temps, axis=0))} unique temperature combinations.")
            plot_model_predictions_by_temp(y_true, y_pred, temps)
        else:
            print("Plotting without temperatures - using regular scatter plot.")
            plot_model_predictions(y_true, y_pred)
            
        if title:
            plt.title(title)
        else:
            plt.title('XGBoost Predictions vs Actual Values')
        
        # Save plot
        plt.savefig(plots_dir / output_name)
        plt.close()
        print(f"Predictions plot saved to {plots_dir / output_name}")

    def evaluate_by_range(self, X_test, y_test):
        """Evaluate predictions across different ranges of solid fraction"""
        dtest = xgb.DMatrix(X_test)
        predictions = self.model.predict(dtest)
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
    """
    --split: Choose which train/test split to use (any split from TRAIN_TEST_SPLITS)​

    --mode: Select "validation" or "test" mode​

    --lr: Set the learning rate, 0.01 defualt​

    --depth: Set the maximum tree depth, 6 default​

    --boost-rounds: Set the number of boosting rounds, 1000 defualt
    
    To run on the 2000K test data:
    python gbm.py --mode test --split train_2500_val_3500_test_2000
    
    To run with the 300K/2000K in training
    python gbm.py --mode test --split bring_in_300_2000
    To try different learning rates:
    python gbm.py --mode test --split bring_in_300_2000 --lr 0.005
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost model for XRD analysis.")
    parser.add_argument("--split", type=str, default="original", choices=TRAIN_TEST_SPLITS.keys(), 
                        help="Training split to use (default: original)")
    parser.add_argument("--mode", type=str, default="validation", choices=["validation", "test"],
                        help="Evaluation mode: 'validation' or 'test' (default: validation)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--depth", type=int, default=6, help="Maximum tree depth (default: 6)")
    parser.add_argument("--boost-rounds", type=int, default=1000, help="Number of boosting rounds (default: 1000)")
    args = parser.parse_args()
    
    # Use specified split
    split = TRAIN_TEST_SPLITS[args.split]
    print(f"Using {args.split} split:")
    print(f"  Training data: {split.train_data}")
    print(f"  Validation data: {split.validation_data}")
    print(f"  Test data: {split.test_data}")
    
    # Set model parameters
    params = {
        'learning_rate': args.lr,
        'max_depth': args.depth
    }
    
    # Initialize model
    xrd_boost = XRDBoost(params=params)
    xrd_boost.num_boost_rounds = args.boost_rounds
    
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
        
        # Use a small portion of the training data as validation for early stopping
        train_idx = np.random.choice(len(X_train), int(0.9*len(X_train)), replace=False)
        val_idx = np.array([i for i in range(len(X_train)) if i not in train_idx])
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        X_train_subset = X_train[train_idx]
        y_train_subset = y_train[train_idx]
        
        # Train model
        model = xrd_boost.train(X_train_subset, y_train_subset, X_val, y_val)
        
        # Evaluate on test data
        predictions, metrics = xrd_boost.evaluate(X_test, y_test)
        
        # Create descriptive filenames based on the split and mode
        plot_title = f"XGBoost: {args.split} Split (Test Mode)"
        feature_importance_filename = f"xgboost_{args.split}_test_feature_importance.png"
        predictions_filename = f"xgboost_{args.split}_test_predictions.png"
        
        print(f"\nTest Set Performance ({args.split}):")
        print(f"Mean Squared Error: {metrics['mse']:.6f}")
        print(f"Mean Absolute Error: {metrics['mae']:.6f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
        print(f"R² Score: {metrics['r2']:.6f}")
        
        # Plot feature importance
        xrd_boost.plot_feature_importance(output_name=feature_importance_filename)
        
        # Plot predictions
        xrd_boost.plot_predictions(y_test, predictions, test_temps, 
                                  output_name=predictions_filename,
                                  title=plot_title)
        
        # Evaluate by range
        range_results = xrd_boost.evaluate_by_range(X_test, y_test)
        print("\nTest Performance by solid fraction range:")
        for range_name, metrics in range_results.items():
            print(f"\nRange {range_name}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  Number of samples: {metrics['n_samples']}")
            
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
        
        # Train model
        model = xrd_boost.train(X_train, y_train, X_val, y_val)
        
        # Evaluate performance
        predictions, metrics = xrd_boost.evaluate(X_val, y_val)
        
        # Create descriptive filenames based on the split and mode
        plot_title = f"XGBoost: {args.split} Split (Validation Mode)"
        feature_importance_filename = f"xgboost_{args.split}_val_feature_importance.png"
        predictions_filename = f"xgboost_{args.split}_val_predictions.png"
        
        print("\nValidation Performance:")
        print(f"Mean Squared Error: {metrics['mse']:.6f}")
        print(f"Mean Absolute Error: {metrics['mae']:.6f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
        print(f"R² Score: {metrics['r2']:.6f}")
        
        # Plot feature importance
        xrd_boost.plot_feature_importance(output_name=feature_importance_filename)
        
        # Plot predictions
        xrd_boost.plot_predictions(y_val, predictions, val_temps,
                                  output_name=predictions_filename,
                                  title=plot_title)
        
        # Evaluate by range
        range_results = xrd_boost.evaluate_by_range(X_val, y_val)
        print("\nValidation Performance by solid fraction range:")
        for range_name, metrics in range_results.items():
            print(f"\nRange {range_name}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  Number of samples: {metrics['n_samples']}")

if __name__ == "__main__":
    main()
