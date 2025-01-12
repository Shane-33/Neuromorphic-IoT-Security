import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import numpy as np

# Load preprocessed data
print("Loading preprocessed data...")
X_combined = np.load("X_combined.npy")  # Ensure this file exists in the directory
y = np.load("y.npy")  # Ensure this file exists in the directory
print(f"Data loaded. Features Shape: {X_combined.shape}, Labels Shape: {y.shape}")

# Extract EEG features (assuming the last N columns represent EEG data)
# Replace 8562 with the actual number of EEG features
eeg_feature_count = 8562
X_eeg_reshaped = X_combined[:, -eeg_feature_count:]  # Assuming EEG features are at the end
X_eeg_reshaped = X_eeg_reshaped.reshape((X_eeg_reshaped.shape[0], X_eeg_reshaped.shape[1], 1))  # Add channel dimension
print(f"EEG data reshaped for LSTM: {X_eeg_reshaped.shape}")

# Train-test split for EEG data
X_train, X_test, y_train, y_test = train_test_split(X_eeg_reshaped, y, test_size=0.3, random_state=42, stratify=y)

# Define the LSTM model
print("Building LSTM model...")
model = Sequential([
    LSTM(64, input_shape=(X_eeg_reshaped.shape[1], 1), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training LSTM...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
print("Evaluating LSTM...")
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Optionally, save the model
model.save("lstm_model.h5")
print("Model saved as 'lstm_model.h5'.")
