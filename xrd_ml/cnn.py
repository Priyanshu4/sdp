import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
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
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.2),
            
            layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.2),
            
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(512, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
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
                'best_model.keras',  # Updated to new format
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
        
        # Plot training history
        self.plot_training_history(history)
        
        return history

    def plot_training_history(self, history):
        """
        Plot training and validation metrics
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def evaluate_predictions(self, X_test, y_test):
        """
        Evaluate model predictions
        """
        predictions = self.model.predict(X_test)
        mse = np.mean((y_test - predictions.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predictions.flatten()))
        r2 = 1 - (np.sum((y_test - predictions.flatten()) ** 2) / 
                 np.sum((y_test - np.mean(y_test)) ** 2))
        
        # Plot predictions vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
        plt.xlabel('Actual Solid Fraction')
        plt.ylabel('Predicted Solid Fraction')
        plt.title('Predictions vs Actual Values')
        plt.grid(True)
        plt.savefig('predictions_vs_actual.png')
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions
        }

    def evaluate_predictions_by_range(self, X_test, y_test):
        """
        Evaluate predictions across different ranges of solid fraction
        """
        predictions = self.model.predict(X_test)
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        
        results = {}
        for start, end in ranges:
            mask = (y_test >= start) & (y_test < end)
            if np.sum(mask) > 0:
                range_mse = np.mean((y_test[mask] - predictions[mask].flatten()) ** 2)
                range_mae = np.mean(np.abs(y_test[mask] - predictions[mask].flatten()))
                results[f"{start:.1f}-{end:.1f}"] = {
                    "mse": range_mse,
                    "mae": range_mae,
                    "n_samples": np.sum(mask)
                }
        
        return results

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
    print("\nOverall Performance:")
    print(f"Mean Squared Error: {results['mse']}")
    print(f"Root Mean Squared Error: {results['rmse']}")
    print(f"Mean Absolute Error: {results['mae']}")
    print(f"R² Score: {results['r2']}")
    
    # Evaluate by range
    print("\nPerformance by solid fraction range:")
    range_results = xrd_net.evaluate_predictions_by_range(X_val, y_val)
    for range_name, metrics in range_results.items():
        print(f"\nRange {range_name}:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Number of samples: {metrics['n_samples']}")

if __name__ == "__main__":
    main()
