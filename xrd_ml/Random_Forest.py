import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from plotting import plot_model_predictions_by_temp

from train_test_split import (
    load_train_data, 
    load_validation_data, 
    load_test_data_by_temp,  # test data
    get_x_y_as_np_array,
    data_by_temp_to_x_y_np_array   
)

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
        
    def plot_predictions_by_temp(self, y_true, y_pred, temps):
        plt.figure(figsize=(8, 6))
    
        plot_model_predictions_by_temp(y_true, y_pred, temps)
    
        plt.title('Random Forest: Predictions vs Actual Values (Test Data)')
        plt.grid(True)
        plt.savefig('random_forest_test_predictions_vs_actual.png')
        plt.close()

    def train(self, X_train, y_train, X_val, y_val):
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

    def plot_feature_importance(self, feature_names=None):
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
        plt.savefig('random_forest_feature_importance.png')
        plt.close()

    def plot_predictions(self, y_true, y_pred):
        """Plot predicted vs actual solid fraction values."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        # Plot reference line (perfect prediction)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Solid Fraction')
        plt.ylabel('Predicted Solid Fraction')
        plt.title('Random Forest: Predictions vs Actual Values')
        plt.grid(True)
        plt.savefig('random_forest_predictions_vs_actual.png')
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
    print("Loading training data...")
    train_data = load_train_data()
    print("Loading validation data...")
    validation_data = load_validation_data()
    
    # Convert data to numpy arrays
    X_train, y_train = get_x_y_as_np_array(train_data)
    X_val, y_val = get_x_y_as_np_array(validation_data)
    
    # Initialize and train the Random Forest model
    rf_model = XRDRandomForest()
    rf_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate overall performance on the validation data
    predictions, metrics = rf_model.evaluate(X_val, y_val)
    
    print("\nOverall Performance:")
    print(f"Mean Squared Error: {metrics['mse']:.6f}")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
    print(f"R² Score: {metrics['r2']:.6f}")
    
    # Plot feature importance and predictions vs actual
    rf_model.plot_feature_importance()
    rf_model.plot_predictions(y_val, predictions)
    
    # Evaluate performance across different solid fraction ranges
    range_results = rf_model.evaluate_by_range(X_val, y_val)
    print("\nPerformance by solid fraction range:")
    for range_name, met in range_results.items():
        print(f"\nRange {range_name}:")
        print(f"  MSE: {met['mse']:.6f}")
        print(f"  MAE: {met['mae']:.6f}")
        print(f"  Number of samples: {met['n_samples']}")
        
    # Load test data instead of validation data
    print("Loading test data...")
    test_data_by_temp = load_test_data_by_temp(suppress_load_errors=True)
    
    # Convert data to numpy arrays
    X_train, y_train = get_x_y_as_np_array(train_data)
    
    # Convert test data - this returns both the data and the temperature info
    X_test, y_test, test_temps = data_by_temp_to_x_y_np_array(test_data_by_temp)
    
    # Initialize and train the Random Forest model 
    rf_model = XRDRandomForest()
    rf_model.train(X_train, y_train, None, None)  
    
    # Evaluate on test data
    print("Evaluating on test data (2000K)...")
    predictions, metrics = rf_model.evaluate(X_test, y_test)
    
    print("\nTest Data Performance:")
    print(f"Mean Squared Error: {metrics['mse']:.6f}")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
    print(f"R² Score: {metrics['r2']:.6f}")
    
    # Plot predictions by temperature
    rf_model.plot_predictions_by_temp(y_test, predictions, test_temps)
    
    # Evaluate performance across different solid fraction ranges
    range_results = rf_model.evaluate_by_range(X_test, y_test)
    print("\nPerformance by solid fraction range on test data:")
    for range_name, met in range_results.items():
        print(f"\nRange {range_name}:")
        print(f"  MSE: {met['mse']:.6f}")
        print(f"  MAE: {met['mae']:.6f}")
        print(f"  Number of samples: {met['n_samples']}")

if __name__ == "__main__":
    main()
