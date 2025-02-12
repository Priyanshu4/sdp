import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import os
from train_test_split import load_train_data, load_validation_data, get_x_y_as_np_array

class XRDNet:
    def __init__(self):
        self.model = self.build_model()
        
    def build_model(self):
        """
        Build CNN architecture optimized for XRD pattern analysis with regression output
        """
        model = models.Sequential([
            # Input layer - shape is (125, 1) because XRD patterns have 125 points
            layers.Input(shape=(125, 1)),
            
            # Convolutional blocks
            layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(512, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Dense layers for final prediction
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer - predicts liquid fraction
            layers.Dense(1, activation='linear')
        ])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the model with proper validation data
        """
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        # Add callbacks for training
        callbacks = [
            # Save best model
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            # TensorBoard for visualization
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def evaluate_predictions(self, X_test, y_test):
        """
        Evaluate model predictions
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

def main():
    # Load data using your teammate's functions
    print("Loading training data...")
    train_data = load_train_data()
    
    print("Loading validation data...")
    validation_data = load_validation_data()
    
    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    X_train, y_train = get_x_y_as_np_array(train_data)
    X_val, y_val = get_x_y_as_np_array(validation_data)
    
    # Reshape input data for CNN
    X_train = X_train.reshape(-1, 125, 1)
    X_val = X_val.reshape(-1, 125, 1)
    
    # Initialize and train model
    print("Initializing model...")
    xrd_net = XRDNet()
    
    print("Training model...")
    history = xrd_net.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("Evaluating model...")
    results = xrd_net.evaluate_predictions(X_val, y_val)
    print(f"Mean Squared Error: {results['mse']}")
    print(f"Mean Absolute Error: {results['mae']}")
    print(f"R² Score: {results['r2']}")

if __name__ == "__main__":
    main()
