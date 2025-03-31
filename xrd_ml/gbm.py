import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self):
        self.model = None
        self.num_boost_rounds = 1000  # Separate parameter to avoid warning
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
            num_boost_round=self.num_boost_rounds,  # Use separate parameter
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

    def plot_feature_importance(self, feature_names=None):
        """Plot feature importance scores"""
        importance_type = 'weight'  # Can also use 'gain' or 'cover'
        scores = self.model.get_score(importance_type=importance_type)
        
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(self.model, importance_type=importance_type)
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        
        # Save plot directly instead of using the helper function
        plt.savefig(plots_dir / 'xgboost_feature_importance.png')
        plt.close()
        print(f"Feature importance plot saved to {plots_dir / 'xgboost_feature_importance.png'}")

    def plot_predictions(self, y_true, y_pred, temps=None):
        """
        Plot predicted vs actual values with temperature information
        
        Parameters:
            y_true: True values
            y_pred: Predicted values
            temps: Temperature information for each data point
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
            
        plt.title('XGBoost Predictions vs Actual Values (Best Model)')
        
        # Save plot directly instead of using the helper function
        plt.savefig(plots_dir / 'xgboost_predictions_vs_actual.png')
        plt.close()
        print(f"Predictions plot saved to {plots_dir / 'xgboost_predictions_vs_actual.png'}")

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

    # Use the train_2500_val_3500_test_2000 split, to use the old one, set old = 1 and comment everything before the if
    old = 0
    
    if old == 1:   
        # Load data
        print("Loading training data...")
        train_data = load_train_data(suppress_load_errors=True)
        
        print("Loading validation data with temperature information...")
        validation_data_by_temp = load_validation_data_by_temp(suppress_load_errors=True)
        
        # Convert to numpy arrays
        X_train, y_train = get_x_y_as_np_array(train_data)
        
        # Get validation data with temperature information
        X_val, y_val, val_temps = data_by_temp_to_x_y_np_array(validation_data_by_temp)
        print(f"Loaded {len(X_val)} validation samples with temperature information")
        
        # Initialize and train model
        xrd_boost = XRDBoost()
        model = xrd_boost.train(X_train, y_train, X_val, y_val)
        
        # Evaluate performance
        predictions, metrics = xrd_boost.evaluate(X_val, y_val)
        
        eval_y = y_val
        eval_temps = val_temps
    else:
        # Use the train_2500_val_3500_test_2000 split
        split = TRAIN_TEST_SPLITS["train_2500_val_3500_test_2000"]
        
        # Load combined training and validation data
        print("Loading training data (including validation)...")
        train_data = load_train_data(split=split, suppress_load_errors=True, include_validation_set=True)
        
        # Load test data (2000K)
        print("Loading test data...")
        test_data_by_temp = load_test_data_by_temp(split=split, suppress_load_errors=True)
        
        # Convert to numpy arrays
        X_train, y_train = get_x_y_as_np_array(train_data)
        X_test, y_test, test_temps = data_by_temp_to_x_y_np_array(test_data_by_temp)
        
        # Use a small subset of train data for validation during training
        # (just for early stopping)
        train_idx = np.random.choice(len(X_train), int(0.9*len(X_train)), replace=False)
        val_idx = np.array([i for i in range(len(X_train)) if i not in train_idx])
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        X_train_subset = X_train[train_idx]
        y_train_subset = y_train[train_idx]
        
        # Initialize and train model
        xrd_boost = XRDBoost()
        model = xrd_boost.train(X_train_subset, y_train_subset, X_val, y_val)
        
        # Evaluate on test data
        predictions, metrics = xrd_boost.evaluate(X_test, y_test)
        
        eval_y = y_test
        eval_temps = test_temps
    
    print("\nOverall Performance:")
    print(f"Mean Squared Error: {metrics['mse']:.6f}")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
    print(f"R² Score: {metrics['r2']:.6f}")
    
    # Plot feature importance
    xrd_boost.plot_feature_importance()
    
    # Plot predictions with temperature information
    print("Plotting predictions with temperature coloring...")
    xrd_boost.plot_predictions(eval_y, predictions, eval_temps)
    
    # Evaluate by range
    range_results = xrd_boost.evaluate_by_range(X_test if old == 0 else X_val, eval_y)
    print("\nPerformance by solid fraction range:")
    for range_name, metrics in range_results.items():
        print(f"\nRange {range_name}:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Number of samples: {metrics['n_samples']}")

if __name__ == "__main__":
    main()
