import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import os
from sklearn.model_selection import train_test_split


class XRDNet:
    def __init__(self):
        self.model = self.build_model()
        
    def build_model(self):
        """
        Build CNN architecture optimized for XRD pattern analysis with regression output
        Key changes:
        - Final layer now has 1 unit with linear activation (for regression)
        - Deeper architecture for better feature extraction
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(125, 1)),
            
            # First convolutional block
            layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Second convolutional block
            layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Third convolutional block
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth convolutional block
            layers.Conv1D(512, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer - single unit for regression
            layers.Dense(1, activation='linear')  # Linear activation for regression
        ])
        return model

    def load_dataset(self, base_path):
        """
        Load XRD patterns and estimate liquid fraction based on position and temperature
        """
        X = []  # XRD patterns
        y = []  # Liquid fraction values
        
        for temp_dir in sorted(glob.glob(os.path.join(base_path, "*Kelvin"))):
            initial_temp = float(os.path.basename(temp_dir).split('_')[0])
            
            for melt_dir in glob.glob(os.path.join(temp_dir, "*Kelvin")):
                melt_temp = float(os.path.basename(melt_dir))
                
                # Process each timestep
                for timestep in range(0, 100501, 1500):
                    for bin_num in range(1, 6):
                        file_path = os.path.join(melt_dir, f"{timestep}.{bin_num}.hist.xrd")
                        if os.path.exists(file_path):
                            pattern = self.load_and_preprocess_data(file_path)
                            if pattern is not None:
                                X.append(pattern)
                                
                                # Estimate liquid fraction based on bin position and temperatures
                                liquid_fraction = self.estimate_liquid_fraction(
                                    bin_num, 
                                    initial_temp, 
                                    melt_temp, 
                                    timestep
                                )
                                y.append(liquid_fraction)
        
        return np.array(X), np.array(y)

    def estimate_liquid_fraction(self, bin_num, initial_temp, melt_temp, timestep):
        """
        Estimate liquid fraction based on position and conditions
        This is a simplified model - you may want to adjust based on your physics
        """
        # Base estimate on bin position
        if bin_num in [1, 5]:  # Top and bottom bins
            base_fraction = 0.0
        elif bin_num == 3:  # Middle bin
            base_fraction = 1.0
        else:  # Transition bins (2 and 4)
            base_fraction = 0.5
            
        # Temperature factor - based on melting point approach
        temp_ratio = initial_temp / melt_temp
        temp_factor = np.clip(temp_ratio, 0, 1)
    
        # Combine factors with physics-based weighting
        liquid_fraction = base_fraction * temp_factor
    
        return np.clip(liquid_fraction, 0, 1)

    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess a single XRD pattern
        """
        try:
            data = np.loadtxt(file_path, skiprows=4)
            intensities = data[:, 1]
            # Normalize intensities
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            return intensities.reshape(-1, 1)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def train(self, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the model with regression-appropriate loss and metrics
        """
        self.model.compile(
            optimizer='adam',
            loss='mse',  # Mean squared error for regression
            metrics=['mae', 'mse']  # Track both mean absolute error and mean squared error
        )
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history

    def evaluate_predictions(self, X_test, y_test):
        """
        Evaluate model predictions with regression metrics
        """
        predictions = self.model.predict(X_test)
        mse = np.mean((y_test - predictions.flatten()) ** 2)
        mae = np.mean(np.abs(y_test - predictions.flatten()))
        r2 = 1 - (np.sum((y_test - predictions.flatten()) ** 2) / 
                 np.sum((y_test - np.mean(y_test)) ** 2))
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions
        }


# Initialize and train model
xrd_net = XRDNet()
base_path = "/gpfs/sharedfs1/MD-XRD-ML/02_Processed-Data"  #this needs to be modified to select the data
X, y = xrd_net.load_dataset(base_path)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #0.3 test_size for a 70/30 split

# Train model
history = xrd_net.train(X_train, y_train)

# Evaluate
results = xrd_net.evaluate_predictions(X_test, y_test)

# Printing out results
print(f"Mean Squared Error: {results['mse']}")
print(f"Mean Absolute Error: {results['mae']}")
print(f"R² Score: {results['r2']}")
