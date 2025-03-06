"""
This script uses keras tuner to search for optimal hyperparameters for the CNN model
"""

from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt
from train_test_split import load_train_data, load_validation_data, get_x_y_as_np_array


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

    # Initialize Hyperband tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=50,   # maximum epochs per model
        factor=3,        # reduction factor for resource allocation
        directory='autotuner_dir',
        project_name='xrdnet_tuning'
    )


    # Load and preprocess the data
    print("Loading training data...")
    train_data = load_train_data(suppress_load_errors=True)
    print("Loading validation data...")
    validation_data = load_validation_data(suppress_load_errors=True)

    print("Converting to numpy arrays...")
    X_train, y_train = get_x_y_as_np_array(train_data)
    X_val, y_val = get_x_y_as_np_array(validation_data)

    # Reshape the input data for the CNN: (samples, 125, 1)
    X_train = X_train.reshape(-1, 125, 1)
    X_val = X_val.reshape(-1, 125, 1)

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
    print(f"Evaluation results - Loss (MSE): {results[0]}, MAE: {results[1]}")

    # Plot the best model and save to file
    keras.utils.plot_model(best_model, show_shapes=True, show_layer_names=True)
    plt.savefig('automl_cnn_best_model.png')

    # Save the best model to file
    best_model.save('automl_cnn_best_model.h5')

if __name__ == "__main__":
    main()