from argparse import ArgumentParser
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting import (
    plot_model_predictions_by_temp,
    save_plot,
    set_plots_subdirectory
)
from train_test_split import (
    load_train_data, 
    load_test_data_by_temp,
    get_x_y_as_np_array,
    data_by_temp_to_x_y_np_array,
    TRAIN_TEST_SPLITS)
from imbalance import resample_dataset_from_binned_solid_fractions

if __name__ == "__main__":
    # Parse command line argument to determine the TRAIN_TEST_SPLIT
    parser = ArgumentParser(description="Train an SVR model with hyperparameters tuned on validation data.")
    parser.add_argument(
        "--split",
        type=str,
        default="train_2000_val_2500_test_3500",
        choices=TRAIN_TEST_SPLITS.keys(),
        help="Specify the train-test split to use (keys from train_test_split.py.TRAIN_TEST_SPLIT).",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Whether to balance the train dataset with resampling",
    )
    args = parser.parse_args()
    print(f"Using train test split: {args.split}")
    split = TRAIN_TEST_SPLITS[args.split]
 
    name = f"svr_cv_gridsearch_{args.split}_split"
    if args.balance:
        name += "_balanced"
    set_plots_subdirectory(name, add_timestamp=True)


    print("Loading dataset...")
    train = load_train_data(split=split, suppress_load_errors=True, include_validation_set=True)
    test = load_test_data_by_temp(split=split, suppress_load_errors=True)
    
    train_x, train_y = get_x_y_as_np_array(train)
    test_x, test_y, test_temps = data_by_temp_to_x_y_np_array(test)
    
    n_samples, n_features = train_x.shape
    print(f"Number of features: {n_features}")
    print(f"Number of training samples: {n_samples}")
    print(f"Number of testing samples: {test_x.shape[0]}")

    if args.balance:
        print("Resampling the training dataset to balance it...")
        train_x, train_y = resample_dataset_from_binned_solid_fractions(
            data=train_x,
            solid_fractions=train_y,
            n_bins=20,
            bin_undersampling_threshold=0.8,
            oversample=False,
            random_seed=42
        )
        print(f"Number of training samples after balancing: {train_x.shape[0]}")

    # scikit-learn uses this value when we set gamma='scale'
    gamma_scale = 1 / (n_features * np.var(train_x))
    
    # Define the parameter grid to search
    param_grid = {
        'C': [0.1, 1, 10, 100, 500, 1000],
        'gamma': sorted([gamma_scale, 0.01, 0.1, 1, 10, 100]),
        'epsilon': [0.001, 0.005, 0.01, 0.1, 1]
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
    
    # Test the best model 
    print("\Testing the best model...")
    predictions = best_model.predict(test_x)
    mse = mean_squared_error(test_y, predictions)
    mae = mean_absolute_error(test_y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_y, predictions)
    
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R^2 Score (r2_score): {r2:.6f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    plot_model_predictions_by_temp(test_y, predictions, test_temps)
    plt.title('SVR Predictions vs Actual Values (Best Model)')
    save_plot('svr_cv_gridsearch_best_predictions_vs_actual.png')
    
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
    save_plot('svr_cv_gridsearch.png')
    
    print("Visualization plots saved to disk")
