from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_test_split import (
    load_train_data, 
    load_validation_data,
    get_x_y_as_np_array)
from plotting import plot_solid_fraction_distribution

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
    
    # Define the parameter grid to search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 0.2]
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
    
    # Create results DataFrame for all tested parameters
    results = pd.DataFrame(grid_search.cv_results_)
    
    # plot solid fraction distribution for train and val
    plt.figure()
    plot_solid_fraction_distribution(train, bins=20)
    plt.title("Train Data Solid Fraction Distribution")
    plt.savefig("train_solid_fraction_distribution.png")
    
    plt.figure()
    plot_solid_fraction_distribution(validation, bins=20)
    plt.title("Validation Data Solid Fraction Distribution")
    plt.savefig("validation_solid_fraction_distribution.png")
    
    # Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(validation_y, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Solid Fraction')
    plt.ylabel('Predicted Solid Fraction')
    plt.title('SVR Predictions vs Actual Values (Best Model)')
    plt.grid(True)
    plt.savefig('svr_best_predictions_vs_actual.png')
    
    # Visualize hyperparameter performance
    plt.figure(figsize=(15, 10))
    
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
    plt.subplot(1, 2, 1)
    heatmap = plt.imshow(-pivot_table, cmap='viridis', aspect='auto')
    plt.colorbar(heatmap, label='Negative MSE')
    plt.title('Hyperparameter Performance (C vs gamma)')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    
    # Plot convergence of top models
    plt.subplot(1, 2, 2)
    top_results = results.sort_values('mean_test_score', ascending=False).head(10)
    x_values = range(1, len(top_results) + 1)
    plt.plot(x_values, -top_results['mean_test_score'], 'o-')
    plt.title('Top 10 Models Performance')
    plt.xlabel('Model Rank')
    plt.ylabel('MSE (Cross-Validation)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png')
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'C': results['param_C'],
        'gamma': results['param_gamma'],
        'epsilon': results['param_epsilon'],
        'mean_test_score': -results['mean_test_score'],
        'std_test_score': results['std_test_score'],
        'rank_test_score': results['rank_test_score']
    })
    results_df.sort_values('mean_test_score').to_csv('svr_grid_search_results.csv', index=False)
    
    print("\nGrid search results saved to 'svr_grid_search_results.csv'")
    print("Visualization plots saved to disk")
