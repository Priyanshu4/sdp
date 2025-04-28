from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from plotting import (
    plot_model_predictions_by_temp,
    set_plots_subdirectory,
    save_plot
)
from train_test_split import (
    load_train_data,
    load_validation_data_by_temp,   
    load_test_data_by_temp,
    get_x_y_as_np_array,
    data_by_temp_to_x_y_np_array,
    TRAIN_TEST_SPLITS,
)
from imbalance import resample_dataset_from_binned_solid_fractions

if __name__ == "__main__":

    # Parse command line argument to determine the TRAIN_TEST_SPLIT
    parser = ArgumentParser(description="Train an SVR model with hyperparameters tuned on validation data.")
    parser.add_argument(
        "--split",
        type=str,
        default="original",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Whether to balance the train dataset with resampling",
    )
    args = parser.parse_args()
    if args.split not in TRAIN_TEST_SPLITS:
        raise ValueError(f"Invalid train_test_split value. Choose from {TRAIN_TEST_SPLITS.keys()}.")
    print(f"Using train_test_split: {args.split}")
    split = TRAIN_TEST_SPLITS[args.split]

    name = f"svr_gridsearch_{args.split}_split"
    if args.balance:
        name += "_balanced"
    set_plots_subdirectory(name, add_timestamp=True)

    print("Loading dataset...")
    train = load_train_data(split=split, suppress_load_errors=True, include_validation_set=True)
    validation = load_validation_data_by_temp(split=split, suppress_load_errors=True)
    test = load_test_data_by_temp(split=split, suppress_load_errors=True)
    
    # Convert datasets to numpy arrays
    train_x, train_y = get_x_y_as_np_array(train)
    validation_x, validation_y, validation_temps = data_by_temp_to_x_y_np_array(validation)
    test_x, test_y, test_temps = data_by_temp_to_x_y_np_array(test)
    
    n_samples, n_features = train_x.shape
    print(f"Number of features: {n_features}")
    print(f"Number of training samples: {n_samples}")
    print(f"Number of validation samples: {validation_x.shape[0]}")
    print(f"Number of testing samples: {test_x.shape[0]}")

    if args.balance == 1:
        print("Resampling the training dataset to balance it...")
        print("Using resample_dataset_from_binned_solid_fractions")
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
    
    # Manual grid search without cross-validation
    best_score = np.inf
    best_params = {}
    best_model = None
    results_list = []
    
    print("Starting grid search (no cross-validation...)")
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            for epsilon in param_grid['epsilon']:
                # Create and train the model
                model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                model.fit(train_x, train_y)
                
                # Evaluate on the validation set
                preds = model.predict(validation_x)
                mse = mean_squared_error(validation_y, preds)
                
                # Save the result for this parameter combination
                results_list.append({
                    'param_C': C,
                    'param_gamma': gamma,
                    'param_epsilon': epsilon,
                    'validation_mse': mse
                })
                
                # Track the best model
                if mse < best_score:
                    best_score = mse
                    best_params = {'C': C, 'gamma': gamma, 'epsilon': epsilon}
                    best_model = model
    
    print("\nBest parameters found:")
    print(best_params)
    print(f"Best Score (Validation MSE): {best_score:.6f}")

    # get validation r^2 score
    print("\nEvaluating the best model on the validation set...")
    validation_preds = best_model.predict(validation_x)
    validation_mse = mean_squared_error(validation_y, validation_preds)
    validation_mae = mean_absolute_error(validation_y, validation_preds)
    validation_rmse = np.sqrt(validation_mse)
    validation_r2 = r2_score(validation_y, validation_preds)
    
    print(f"Validation Mean Squared Error (MSE): {validation_mse:.6f}")
    print(f"Validation Mean Absolute Error (MAE): {validation_mae:.6f}")
    print(f"Validation Root Mean Squared Error (RMSE): {validation_rmse:.6f}")
    print(f"Validation R^2 Score: {validation_r2:.6f}")
    
    # Plot validation predictions vs actual
    plt.figure(figsize=(8, 6))
    plot_model_predictions_by_temp(validation_y, validation_preds, validation_temps)
    plt.title('SVR Predictions vs Actual Values (Validation Set)')
    save_plot('svr_gridsearch_validation_predictions_vs_actual.png')

    # Test the best model 
    print("\nTesting the best model on the test set...")
    predictions = best_model.predict(test_x)
    mse = mean_squared_error(test_y, predictions)
    mae = mean_absolute_error(test_y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_y, predictions)
    
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R^2 Score: {r2:.6f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    plot_model_predictions_by_temp(test_y, predictions, test_temps)
    plt.title('SVR Predictions vs Actual Values (Best Model)')
    save_plot('svr_gridsearch_best_predictions_vs_actual.png')
    
    # Create results DataFrame for all tested parameters
    results_df = pd.DataFrame(results_list)
    
    # Group by C and gamma, and select the best epsilon for each combination
    grouped_results = results_df.groupby(['param_C', 'param_gamma']).apply(
        lambda x: x.loc[x['validation_mse'].idxmin()]
    ).reset_index(drop=True)
    
    # Pivot table for heatmap (using validation MSE)
    pivot_table = grouped_results.pivot_table(
        values='validation_mse',
        index='param_C', 
        columns='param_gamma'
    )
    
    # Plot heatmap of the validation MSE
    plt.figure()
    heatmap = plt.imshow(pivot_table, cmap='viridis', aspect='auto')
    plt.colorbar(heatmap, label='Validation MSE')
    plt.title('Hyperparameter Performance (C vs gamma)')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.xticks(range(len(pivot_table.columns)), [round(x, 2) for x in pivot_table.columns])
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    save_plot('svr_gridsearch.png')
    
    print("Visualization plots saved to disk")
