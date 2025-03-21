import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from plotting import (
    plot_model_predictions,
    save_plot
)
from train_test_split import (
    load_train_data, 
    load_validation_data,
    get_x_y_as_np_array
)
from plotting import plot_solid_fraction_distribution

class XRDBoost:
    def __init__(self):
        self.model = None
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 1000,
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
            num_boost_round=self.params['n_estimators'],
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

    def plot_predictions(self, y_true, y_pred):
        """Plot predicted vs actual values"""
        plt.figure(figsize=(8, 6))
        plot_model_predictions(y_true, y_pred)
        plt.title('XGBoost: Predictions vs Actual Values')
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
    print("Loading validation data...")
    validation_data = load_validation_data()
    
    # Convert to numpy arrays
    X_train, y_train = get_x_y_as_np_array(train_data)
    X_val, y_val = get_x_y_as_np_array(validation_data)
    
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
    
    # Plot predictions
    xrd_boost.plot_predictions(y_val, predictions)
    
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
