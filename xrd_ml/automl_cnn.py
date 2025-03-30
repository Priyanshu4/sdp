"""
This script uses keras tuner to search for optimal hyperparameters for the CNN model
"""

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras_tuner as kt
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from train_test_split import (
    TRAIN_TEST_SPLITS,
    load_train_data, 
    load_validation_data,
    load_test_data_by_temp,
    get_x_y_as_np_array,
    data_by_temp_to_x_y_np_array
)
from plotting import (
    plot_model_predictions_by_temp,
    set_plots_subdirectory,
    get_plots_subdirectory,
    save_plot
)
from imbalance import resample_dataset_from_binned_solid_fractions


def build_model(hp):
    """ 
    Build a CNN given hyperaparameters hp
    Used by keras tuner to search for optimal hyperparameters
    """
    model = keras.Sequential()

    # Input layer: XRD patterns with 125 points and 1 channel
    model.add(layers.Input(shape=(125, 1)))

    # First convolutional and pooling block
    conv_filters = hp.Choice('conv_1_filters', values=[32, 64, 128])
    conv_kernel = hp.Choice('conv_1_kernel', values=[3, 5, 7])
    pool_size = hp.Choice('pool_1_size', values=[2, 4])
    model.add(layers.Conv1D(conv_filters, kernel_size=conv_kernel, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    dropout_rate = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(layers.Dropout(dropout_rate))

    # Second convolutional and pooling block
    conv_filters = hp.Choice('conv_2_filters', values=[64, 128, 256])
    conv_kernel = hp.Choice('conv_2_kernel', values=[3, 5])
    model.add(layers.Conv1D(conv_filters, kernel_size=conv_kernel, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    pool_size = hp.Choice('pool_2_size', values=[2, 4])
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    dropout_rate = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    model.add(layers.Dropout(dropout_rate))

    # Flatten and dense layers
    model.add(layers.Flatten())
    dense_units = hp.Int('dense_units', min_value=256, max_value=1024, step=128)
    model.add(layers.Dense(dense_units, activation='relu'))
    dropout_rate = hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1)
    model.add(layers.Dropout(dropout_rate))

    # Output layer for regression (predicting liquid fraction)
    final_activation = hp.Choice('final_activation', values=['linear', 'relu', 'sigmoid'])
    model.add(layers.Dense(1, activation=final_activation))


    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae'])

    return model       

def main():

    # Parse command line argument to determine the TRAIN_TEST_SPLIT
    parser = ArgumentParser(description="Train an SVR model with hyperparameters tuned on validation data.")
    parser.add_argument(
        "--train_test_split",
        type=str,
        default="original",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Whether to balance the train dataset with resampling."
    )
    args = parser.parse_args()
    if args.train_test_split not in TRAIN_TEST_SPLITS:
        raise ValueError(f"Invalid train_test_split value. Choose from {TRAIN_TEST_SPLITS.keys()}.")
    print(f"Using train_test_split: {args.train_test_split}")
    split = TRAIN_TEST_SPLITS[args.train_test_split]

    name = f"automl_cnn_{args.train_test_split}_split"
    if args.balance:
        name += "_balanced"
    set_plots_subdirectory(name, add_timestamp=True)

    # Initialize Hyperband tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=50,   # maximum epochs per model
        factor=3,        # reduction factor for resource allocation
        directory='autotuner_dir',
        project_name=name,
    )

    # Load and preprocess the data
    print("Loading training data...")
    train_data = load_train_data(split=split, suppress_load_errors=True)
    X_train, y_train = get_x_y_as_np_array(train_data)

    print("Loading validation data...")
    validation_data = load_validation_data(split=split, suppress_load_errors=True)
    X_val, y_val = get_x_y_as_np_array(validation_data)

    # Load the test data and evaluate the best model
    print("Loading test data...")
    test_data = load_test_data_by_temp(split=split)
    X_test, Y_test, temps_test = data_by_temp_to_x_y_np_array(test_data)

    # Reshape the input data for the CNN: (samples, 125, 1)
    X_test = X_test.reshape(-1, 125, 1)
    X_train = X_train.reshape(-1, 125, 1)
    X_val = X_val.reshape(-1, 125, 1)

    if args.balance:
        print("Resampling the training dataset to balance it...")
        X_train, Y_train = resample_dataset_from_binned_solid_fractions(
            data=X_train,
            solid_fractions=Y_train,
            n_bins=20,
            n_samples_per_bin=1000,
            oversample=True,
            random_seed=42
        )
        print(f"Number of training samples after balancing: {X_train.shape[0]}")
    
    # Start the hyperparameter search using Hyperband
    tuner.search(X_train, y_train, 
                validation_data=(X_val, y_val), 
                epochs=50, 
                batch_size=32)

    # Retrieve the best model and hyperparameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found:")
    print(best_hyperparameters.values)

    # Evaluate the best model on validation data
    results = best_model.evaluate(X_val, y_val)
    print(f"Evaluation results on validation data - Loss (MSE): {results[0]}, MAE: {results[1]}")

    # Plot the best model and save to file
    print("Plotting the best model...")
    plot_file = 'automl_cnn_best_model.png'
    keras.utils.plot_model(best_model, show_shapes=True, show_layer_names=True, to_file = get_plots_subdirectory() / plot_file)
    print(f"Plot saved to {plot_file}")

    # Evaluate the best model on test data
    print("Evaluation on test data...")
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(Y_test, predictions)
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")

    plt.figure()
    plt.title("Predictions of the Best CNN on Test Data")
    plot_model_predictions_by_temp(Y_test, predictions, temps_test)
    save_plot('automl_cnn_test_predictions.png')
    print("Plot saved to automl_cnn_test_predictions.png")

    # Save the best model to file
    best_model.save(get_plots_subdirectory() / 'automl_cnn_best_model.h5')

if __name__ == "__main__":
    main()