import argparse
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import plotting functions and data loading utilities
from plotting import (
    plot_model_predictions_by_temp,
    save_plot,
    set_plots_subdirectory,
)
from train_test_split import (
    load_train_data, 
    load_test_data_by_temp,
    load_validation_data_by_temp,
    get_x_y_as_np_array,
    data_by_temp_to_x_y_np_array,
    TRAIN_TEST_SPLITS,
)
from imbalance import resample_dataset_from_binned_solid_fractions

def parse_gamma(value):
    """
    Parse the gamma argument. If the value is 'scale' or 'auto', return as a string.
    Otherwise, try converting it to a float.
    """
    lower_value = value.lower()
    if lower_value in ["scale", "auto"]:
        return lower_value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("gamma must be a float, 'scale', or 'auto'.")

def main():
    # Parse command line arguments for hyperparameters and evaluation mode.
    parser = argparse.ArgumentParser(description="Train and evaluate an SVR model using command-line specified hyperparameters.")
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
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter (default: 1.0)")
    parser.add_argument("--gamma", type=parse_gamma, default="scale", help="Kernel coefficient (default: 'scale'). Accepts a float, 'scale', or 'auto'.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon in the epsilon-SVR model (default: 0.1)")
    parser.add_argument("--test", action="store_true", help="If provided, train on the full training data (including validation) and evaluate on the test dataset. Otherwise, train on the train data (excluding validation) and evaluate on the validation dataset.")
    
    args = parser.parse_args()
    print(f"Using train test split: {args.split}")
    split = TRAIN_TEST_SPLITS[args.split]
 
    name = f"svr_{args.split}_split"
    if args.balance:
        name += "_balanced"
    set_plots_subdirectory(name, add_timestamp=True)

    # Load the appropriate training and evaluation datasets based on the --test flag
    if args.test:
        print("Using full training data (including validation) and test dataset for evaluation.")
        print("This is for final evaluation purposes only.")
        print("Loading data...")
        train = load_train_data(split=split, suppress_load_errors=True, include_validation_set=True)
        eval_data = load_test_data_by_temp(split=split, suppress_load_errors=True)
    else:
        print("Using train data (excluding validation) and validation dataset for evaluation.")
        print("Loading data...")
        train = load_train_data(split=split, suppress_load_errors=True, include_validation_set=False)
        eval_data = load_validation_data_by_temp(split=split, suppress_load_errors=True)

    # Convert datasets to numpy arrays
    train_x, train_y = get_x_y_as_np_array(train)
    eval_x, eval_y, eval_temps = data_by_temp_to_x_y_np_array(eval_data)

    n_train_samples, n_features = train_x.shape
    n_eval_samples = eval_x.shape[0]
    print(f"Number of training samples: {n_train_samples}")
    print(f"Number of features: {n_features}")
    print(f"Number of evaluation samples: {n_eval_samples}")

    # Optionally balance the training dataset
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

    # Create and train the SVR model using provided hyperparameters.
    print("\nTraining the SVR model with parameters:")
    print(f"  C: {args.C}")
    print(f"  gamma: {args.gamma}")
    print(f"  epsilon: {args.epsilon}")
    model = SVR(kernel='rbf', C=args.C, gamma=args.gamma, epsilon=args.epsilon)
    model.fit(train_x, train_y)

    # Evaluate the model on the evaluation dataset.
    print("\nEvaluating the model...")
    predictions = model.predict(eval_x)
    mse = mean_squared_error(eval_y, predictions)
    mae = mean_absolute_error(eval_y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(eval_y, predictions)

    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R^2 Score: {r2:.6f}")

    # Plot predictions vs actual values.
    plt.figure(figsize=(8, 6))
    plot_model_predictions_by_temp(eval_y, predictions, eval_temps)
    plt.title('SVR Predictions vs Actual Values')
    
    # Save plot with an appropriate file name.
    if args.test:
        plot_filename = 'svr_test_predictions_vs_actual.png'
    else:
        plot_filename = 'svr_validation_predictions_vs_actual.png'
    save_plot(plot_filename)
    
    print(f"Results and plot saved to disk as '{plot_filename}'.")

if __name__ == "__main__":
    main()
