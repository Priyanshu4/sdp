from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
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

    model = SVR(
        kernel = 'rbf',
        gamma = 'scale',
        C = 1.0,
        verbose=True
    )

    # Fit the model to the training data
    print("Training the model...")
    model.fit(train_x, train_y)
    print()

    # Test the model on the validation data
    print("Validating the model...")
    predictions = model.predict(validation_x)
    mse = mean_squared_error(validation_y, predictions)
    mae = mean_absolute_error(validation_y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(validation_y, predictions)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score (r2_score): {r2}")

