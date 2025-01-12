import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
file_path = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset_500MB.csv"
data = pd.read_csv(file_path)

# Handle missing values using the recommended method
data.fillna(method='ffill', inplace=True)  # Forward fill
data.fillna(method='bfill', inplace=True)  # Backward fill

# Separate IoT features, EEG spike trains, and labels
# Fix the typo: "columpythonns" -> "columns"
X_iot = data.drop(columns=['label', 'type', 'spike_train'], errors='ignore')

# Safely evaluate and reshape the spike_train column
X_eeg = np.array([np.array(eval(spike)) for spike in data['spike_train']])
y = data['label']

# Encode categorical features for IoT data
categorical_columns = X_iot.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X_iot[col] = le.fit_transform(X_iot[col])

# Scale IoT features
scaler = StandardScaler()
X_iot_scaled = scaler.fit_transform(X_iot)

# Reshape X_eeg to align with X_iot_scaled dimensions
# Assuming each EEG spike train has the same length
X_eeg_reshaped = X_eeg.reshape(X_eeg.shape[0], -1)

# Combine IoT and EEG features
X_combined = np.hstack((X_iot_scaled, X_eeg_reshaped))

# Save preprocessed data
np.save("X_combined.npy", X_combined)
np.save("y.npy", y)

print("Preprocessing complete. Combined dataset saved as 'X_combined.npy' and labels saved as 'y.npy'.")
