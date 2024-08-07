import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from data_generation import generate_potential_data

# Check if TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version: ", tf.__version__)

# If the GPU is available, use mixed precision for faster computation
if tf.config.list_physical_devices('GPU'):
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')

def create_model(input_shape, num_eigenvalues):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_eigenvalues, activation='linear', dtype='float32')  # Ensure the output is in float32
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Parameters
num_samples = 500
grid_size = 30
num_eigenvalues = 25  # Number of eigenvalues to predict

# Generate data
X_train, y_train = generate_potential_data(num_samples, grid_size, num_eigenvalues)
X_val, y_val = generate_potential_data(10, grid_size, num_eigenvalues)
X_test, y_test = generate_potential_data(10, grid_size, num_eigenvalues)

# Reshape data for the CNN
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Create and train the model
model = create_model((grid_size, grid_size, 1), num_eigenvalues)
model.summary()

# Fit the model and save the training history
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epoch')
plt.show()

# Evaluate the model on the test set and report the test loss and the mean absolute error
results = model.evaluate(X_test, y_test, verbose=1)
print("Test loss and metrics:", results)

# Test the model on the test set and print predictions and true values
predictions = model.predict(X_test)

print("Predictions:", predictions)
print("True values:", y_test)
