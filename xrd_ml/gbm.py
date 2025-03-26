import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotting import (
    plot_model_predictions,
    plot_model_predictions_by_temp,
    save_plot
)
from train_test_split import (
    load_train_data, 
    load_validation_data,
    get_x_y_as_np_array
)
from plotting import plot_solid_fraction_distribution

# Define the temperature-aware functions
def load_validation_data_by_temp(suppress_load_errors=True):
    """Load validation data and organize it by temperature"""
    from train_test_split import load_validation_data
    data = load_validation_data(suppress_load_errors=suppress_load_errors)
    
    # Group data by temperature and melting temperature
    by_temp = {}
    for idx, row in data.iterrows():
        if 'baseTemp' in row and 'meltTemp' in row:
            temp_key = (row['baseTemp'], row['meltTemp'])
        else:
            # Extract temperature from filename if available
            filename = row.get('filename', '')
            if isinstance(filename, str):
                parts = filename.split('/')
                for part in parts:
                    if part.startswith('0'):  # Folder with temperature info
                        try:
                            base_temp = int(part.split('-')[0].strip('0'))
                            melt_temp = 2500  # Default
                            if 'Kelvin/' in filename:
                                melt_folder = filename.split('Kelvin/')[1].split('/')[0]
                                try:
                                    melt_temp = int(melt_folder.split('-')[0])
                                except:
                                    pass
                            temp_key = (base_temp, melt_temp)
                        except:
                            temp_key = (300, 2500)  # Default fallback
            else:
                temp_key = (300, 2500)  # Default fallback
        
        if temp_key not in by_temp:
            by_temp[temp_key] = []
        by_temp[temp_key].append(row)
    
    return by_temp

def data_by_temp_to_x_y_np_array(data_by_temp):
    """Convert temperature-organized data to numpy arrays for ML"""
    all_x = []
    all_y = []
    all_temps = []
    
    for temp_key, rows in data_by_temp.items():
        df = pd.DataFrame(rows)
        x, y = get_x_y_as_np_array(df)
        all_x.append(x)
        all_y.append(y)
        all_temps.extend([temp_key] * len(y))
    
    return np.vstack(all_x), np.concatenate(all_y), all_temps

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
        save_plot('xgboost_feature_importance.png')

    def plot_predictions(self, y_true, y_pred, temps=None):
        """
        Plot predicted vs actual values with temperature information
        
        Parameters:
            y_true: True values
            y_pred: Predicted values
            temps: Temperature information for each data point
        """
        plt.figure(figsize=(8, 6))
        if temps is not None:
            plot_model_predictions_by_temp(y_true, y_pred, temps)
        else:
            plot_model_predictions(y_true, y_pred)
        plt.title('XGBoost Predictions vs Actual Values (Best Model)')
        save_plot('xgboost_predictions_vs_actual.png')

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
    # Load data
    print("Loading training data...")
    train_data = load_train_data()
    
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
    
    print("\nOverall Performance:")
    print(f"Mean Squared Error: {metrics['mse']:.6f}")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
    print(f"R² Score: {metrics['r2']:.6f}")
    
    # Plot feature importance
    xrd_boost.plot_feature_importance()
    
    # Plot predictions with temperature information
    print("Plotting predictions with temperature coloring...")
    xrd_boost.plot_predictions(y_val, predictions, val_temps)
    
    # Evaluate by range
    range_results = xrd_boost.evaluate_by_range(X_val, y_val)
    print("\nPerformance by solid fraction range:")
    for range_name, metrics in range_results.items():
        print(f"\nRange {range_name}:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Number of samples: {metrics['n_samples']}")

if __name__ == "__main__":
    main()
