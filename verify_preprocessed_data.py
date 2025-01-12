import numpy as np

# Load preprocessed data
X_combined = np.load("X_combined.npy")
y = np.load("y.npy")

# Verify shapes
print("Combined Features Shape:", X_combined.shape)
print("Labels Shape:", y.shape)

# Check for consistency
if len(X_combined) != len(y):
    print("Error: Number of samples in X_combined and y do not match!")
else:
    print("Data verification successful.")
