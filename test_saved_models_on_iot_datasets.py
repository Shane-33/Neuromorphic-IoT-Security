import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from joblib import load

# Spiking Decision Tree Implementation
class SpikingDecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(y) == 0 or depth >= self.max_depth or len(set(y)) == 1:
            return {"label": max(set(y), key=list(y).count) if len(y) > 0 else 0}

        node = {
            "threshold": np.mean(X[:, 0]) + 0.05 * np.std(X[:, 0]),  # Slight bias for SDT
            "feature": 0
        }
        left_mask = X[:, node["feature"]] < node["threshold"]
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {"label": max(set(y), key=list(y).count)}

        node["left"] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x, node):
        if "label" in node:
            return node["label"]
        if x[node["feature"]] < node["threshold"]:
            return self._predict_single(x, node["left"])
        else:
            return self._predict_single(x, node["right"])

# Define SNNFeatureExtractor
class SNNFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNNFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Preprocess data
def preprocess_data(filepath, feature_dim):
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)

    non_numeric_columns = data.select_dtypes(include=['object']).columns
    for col in non_numeric_columns:
        if col == 'label':
            continue
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['label'])
    y = data['label']
    X = data.drop(columns=['label'])

    print("Handling missing values...")
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    if X.shape[1] < feature_dim:
        print(f"Adding {feature_dim - X.shape[1]} missing columns to match feature dimension.")
        X = np.hstack([X, np.zeros((X.shape[0], feature_dim - X.shape[1]))])
    elif X.shape[1] > feature_dim:
        print(f"Trimming {X.shape[1] - feature_dim} extra columns.")
        X = X[:, :feature_dim]

    return X, y

# Evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    print(f"Evaluating {model_name}...")
    predictions = model.predict(X_test)
    print(f"Predictions (first 10): {predictions[:10]}")
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy:.4f}\n")


# Test saved models
def test_models():
    model_folder = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Model"
    dataset_folder = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Train_Test_IoT_dataset"
    dataset_files = [
        "Train_Test_IoT_Fridge.csv",
        "Train_Test_IoT_Garage_Door.csv",
        "Train_Test_IoT_GPS_Tracker.csv",
        "Train_Test_IoT_Modbus.csv",
        "Train_Test_IoT_Motion_Light.csv",
        "Train_Test_IoT_Thermostat.csv",
        "Train_Test_IoT_Weather.csv",
    ]

    # Load feature dimension
    with open(f"{model_folder}/feature_dimension.txt", "r") as f:
        feature_dim = int(f.read())

    print("Loading models...")
    sdt_model = load(f"{model_folder}/sdt_model.pkl", globals())
    print("Loaded Spiking Decision Tree model.")

    rf_model = load(f"{model_folder}/rf_model.pkl")
    print("Loaded Random Forest model.")

    rf_hybrid = load(f"{model_folder}/snn_rf_model.pkl")
    print("Loaded SNN+RF Hybrid model.")

    scaler = load(f"{model_folder}/scaler.pkl")
    print("Loaded Scaler.")

    pca = load(f"{model_folder}/pca.pkl")
    print("Loaded PCA.")

    snn = SNNFeatureExtractor(input_size=feature_dim, hidden_size=128, output_size=64)
    snn.load_state_dict(torch.load(f"{model_folder}/snn_feature_extractor.pth"))
    snn.eval()
    print("Loaded SNN Feature Extractor.")

    for file in dataset_files:
        filepath = f"{dataset_folder}/{file}"
        try:
            X_test, y_test = preprocess_data(filepath, feature_dim)

            evaluate_model(sdt_model, X_test, y_test, "Spiking Decision Tree")
            evaluate_model(rf_model, X_test, y_test, "Random Forest")

            print("Transforming features using SNN...")
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            snn_features_test = snn(X_test_tensor).detach().numpy()

            print("Applying Scaler and PCA...")
            snn_features_test = scaler.transform(snn_features_test)
            snn_features_test = pca.transform(snn_features_test)

            evaluate_model(rf_hybrid, snn_features_test, y_test, "SNN+RF Hybrid")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    test_models()







