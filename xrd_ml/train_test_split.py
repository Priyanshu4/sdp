from load_data import (
    load_processed_data_for_list_of_temps,
    load_processed_data_for_temp_directory)
from typing import Tuple
import pandas as pd
import numpy as np

# (temp, melt_temp) to use in train_data
TRAIN_DATA = [
    (300, 2500),
    (300, 3500),
    (400, 3500),
    (500, 2500),
    (600, 2500),
    (600, 3500),
    (700, 2500),
    (700, 3500),
    (800, 3500),
    (900, 2500),
    (900, 3500),
    (1000, 2500),
    (1100, 2500),
    (1100, 3500),
]

# (temp, melt_temp) to use in validation_data
VALIDATION_DATA = [
    (400, 2500),
    (500, 3500),
    (800, 2500),
    (1000, 3500),
    (1200, 2500),
]

for train_dir in TRAIN_DATA:
    for val_dir in VALIDATION_DATA:
        if train_dir == val_dir:
            raise ValueError(f"Train and validation directories should not have any overlap: {train_dir}")

def load_train_data(suppress_load_errors = False, include_validation_set = False) -> pd.DataFrame:
    """
    Load the training data.

    Parameters:
        suppress_load_errors (bool): Whether to suppress errors during loading
        include_validation_set (bool): Whether to include the validation set in the training data. 
    
    Returns:
        DataFrame: Training data

    See load_processed_data for description of the DataFrame.
    """ 
    if include_validation_set:
        data = TRAIN_DATA + VALIDATION_DATA
    else:
        data = TRAIN_DATA

    return load_processed_data_for_list_of_temps(
        data, 
        suppress_load_errors = suppress_load_errors)
    

def load_validation_data(suppress_load_errors = False) -> pd.DataFrame:
    """
    Load the validation data.

    Parameters:
        suppress_load_errors (bool): Whether to suppress errors during loading
    
    Returns:
        DataFrame: Validation data

    See load_processed_data for description of the DataFrame.
    """ 
    return load_processed_data_for_list_of_temps(
        VALIDATION_DATA, 
        suppress_load_errors = suppress_load_errors)

def load_validation_data_by_temp(suppress_load_errors = False) -> dict[Tuple[int, int], pd.DataFrame]:
    """
    Load the validation data as a dictionary of DataFrames, where the key is the temperature tuple.
    Temperature tuple is (temp, melt_temp).

    Parameters:
        suppress_load_errors (bool): Whether to suppress errors during loading
    
    Returns:
        dict[Tuple[int, int], pd.DataFrame]: Validation data

    See load_processed_data for description of the DataFrame.
    """ 
    validation_data = {}
    for temp, melt_temp in VALIDATION_DATA:
        validation_data[(temp, melt_temp)] = load_processed_data_for_temp_directory(
            temp, melt_temp, suppress_load_errors = suppress_load_errors)
    return validation_data
        
def get_x_y_as_np_array(data: pd.DataFrame,
                        include_missing_Y_data = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get X (hist data) and Y (solid fraction) as numpy arrays.
    If X data is missing for a row, it will be automatically skipped.
    If Y data is missing for a row, it will be skipped only if include_missing_Y_data is False.

    Parameters:
        data (DataFrame): DataFrame containing the data (output of load_train_data or load_validation_data)
        include_missing_Y_data (bool): Whether to include rows with missing (NaN) solidFrac data

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y data
    """
    
    # iterate over the rows of the DataFrame
    X = []
    Y = []

    for _, row in data.iterrows():

        # get the xrd_data
        xrd_data = row["xrd_data"]

        if xrd_data is None:
            # Skip the row if xrd_data is None
            continue

        # extract the Count/Total column
        relative_counts = xrd_data["Count/Total"]

        # convert the relative counts to numpy array of floats
        relative_counts = relative_counts.to_numpy(dtype=np.float32)

        # get the solid fraction and convert it to np.float32
        solid_fraction = row["solidFrac"]

        if np.isnan(solid_fraction) and not include_missing_Y_data:
            # Skip the row if solid fraction is NaN
            continue

        X.append(relative_counts)
        Y.append(solid_fraction)

    X = np.stack(X)
    Y = np.stack(Y)

    return X, Y

def data_by_temp_to_x_y_np_array(data: dict[Tuple[int, int], pd.DataFrame],
                                 include_missing_Y_data = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert the dictionary of DataFrames to an X array, Y array and temperature tuple array.
    Parameters:
        data (dict[Tuple[int, int], pd.DataFrame]): Dictionary of DataFrames containing the data
        include_missing_Y_data (bool): Whether to include rows with missing (NaN) solidFrac data

    Returns:
        X (np.ndarray): X data
        Y (np.ndarray): Y data
        temperatures (np.ndarray): Temperature tuples corresponding to the data (shape n by 2)
    """
    for (temp, melt_temp), df in data.items():
        X, Y = get_x_y_as_np_array(df, include_missing_Y_data)
        if (temp, melt_temp) == VALIDATION_DATA[0]:
            X_all = X
            Y_all = Y
            temperatures = np.array([(temp, melt_temp)] * len(X))
        else:
            X_all = np.concatenate((X_all, X))
            Y_all = np.concatenate((Y_all, Y))
            temperatures = np.concatenate((temperatures, np.array([(temp, melt_temp)] * len(X))))
    return X_all, Y_all, temperatures

if __name__ == "__main__":

    print("Loading training data...")
    train = load_train_data(suppress_load_errors=True)
    print("Train Data:")
    print(train.head())

    print("Loading validation data...")
    validation = load_validation_data(suppress_load_errors=True)
    print("Validation Data:")
    print(validation.head())

    # train as numpy
    train_x, train_y = get_x_y_as_np_array(train)
    print(f"Train X with shape {train_x.shape}:")
    print(train_x)

    print(f"Train Y with shape {train_y.shape}:")
    print(train_y)

    # val as numpy
    validation_x, validation_y = get_x_y_as_np_array(validation)
    print(f"Validation X with shape {validation_x.shape}:")
    print(validation_x)

    print(f"Validation Y with shape {validation_y.shape}:")
    print(validation_y)

    # Check that no two train and validation datapoints are the same
    print('Verifying that train and validation data do not overlap...')
    for i in range(len(train_x)):
        for j in range(len(validation_x)):
            if np.all(train_x[i] == validation_x[j]):
                raise ValueError(f"Train and validation data should not have any overlap: {i}, {j}")




