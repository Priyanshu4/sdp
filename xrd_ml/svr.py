from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting import (
    plot_model_predictions,
    save_plot
)
from train_test_split import (
    load_train_data, 
    load_validation_data,
    get_x_y_as_np_array)

if __name__ == "__main__":
    print("Loading dataset...")
    train = load_train_data(suppress_load_errors=True)
    validation = load_validation_data(suppress_load_errors=True)    
    
    train_x, train_y = get_x_y_as_np_array(train)
    validation_x, validation_y = get_x_y_as_np_array(validation)
    
    n_samples, n_features = train_x.shape
    print(f"Number of features: {n_features}")
    print(f"Number of training samples: {n_samples}")
    print(f"Number of validation samples: {validation_x.shape[0]}")

    # NOTE:
    # In this case, we are treating the validation data as a test set.
    # We are not using the validation data to tune hyperparameters and are only using it to evaluate the model.
    # TODO: Load the test data and use it to evaluate the final model. Combine validation into training data.

    # scikit-learn uses this value when we set gamma='scale'
    gamma_scale = 1 / (n_features * np.var(train_x))
    
    # Define the parameter grid to search
    param_grid = {
        'C': [0.1, 1, 10, 100, 500, 1000],
        'gamma': sorted([gamma_scale, 0.01, 0.1, 1, 10, 100]),
        'epsilon': [0.001, 0.005, 0.01, 0.1, 1]
    }

    param_grid = {
        'C': [500],
        'gamma': sorted([gamma_scale]),
        'epsilon': [0.001]
    }
    
    # Create base SVR model
    svr = SVR(kernel='rbf')
    
    # Set up K-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Set up GridSearchCV
    print("Starting grid search with cross-validation...")
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the grid search to the data
    grid_search.fit(train_x, train_y)
    
    # Print best parameters and score
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {-grid_search.best_score_}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Test the best model on the validation data
    print("\nValidating the best model...")
    predictions = best_model.predict(validation_x)
    mse = mean_squared_error(validation_y, predictions)
    mae = mean_absolute_error(validation_y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(validation_y, predictions)
    
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R^2 Score (r2_score): {r2:.6f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    plot_model_predictions(validation_y, predictions)
    plt.title('SVR Predictions vs Actual Values (Best Model)')
    save_plot('svr_best_predictions_vs_actual.png')
    
    # Create results DataFrame for all tested parameters
    results = pd.DataFrame(grid_search.cv_results_)
    

    # Group by C and gamma, find best epsilon for each combination
    grouped_results = results.groupby(['param_C', 'param_gamma']).apply(
        lambda x: x.loc[x['mean_test_score'].idxmax()]
    ).reset_index(drop=True)
    
    # Pivot table for heatmap
    pivot_table = grouped_results.pivot_table(
        values='mean_test_score',
        index='param_C', 
        columns='param_gamma'
    )
    
    # Plot heatmap
    plt.figure()
    heatmap = plt.imshow(-pivot_table, cmap='viridis', aspect='auto')
    plt.colorbar(heatmap, label='Negative MSE')
    plt.title('Hyperparameter Performance (C vs gamma)')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.xticks(range(len(pivot_table.columns)), [round(x, 2) for x  in pivot_table.columns])
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    save_plot('svr_gridsearch.png')
    
    print("Visualization plots saved to disk")
