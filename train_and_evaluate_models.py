import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
            "threshold": np.mean(X[:, 0]) if len(X) > 0 else 0,
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

# Neural Feature Extractor for SNN+RF Hybrid
class NeuralFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load and Preprocess Data
def load_and_preprocess():
    print("Loading data...")
    X_combined = np.load("X_combined.npy")
    y = np.load("y.npy")

    print("Handling missing values...")
    X_combined = np.nan_to_num(X_combined)

    print("Handling class imbalance...")
    smote = SMOTE(random_state=42)
    X_combined, y = smote.fit_resample(X_combined, y)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Train and Evaluate Models
def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Train Spiking Decision Tree
    print("Training Spiking Decision Tree...")
    sdt = SpikingDecisionTree(max_depth=10)
    sdt.fit(X_train, y_train)
    print("Evaluating Spiking Decision Tree...")
    sdt_predictions = sdt.predict(X_test)
    print("SDT Classification Report:")
    print(classification_report(y_test, sdt_predictions))
    print("SDT Confusion Matrix:")
    cm_sdt = confusion_matrix(y_test, sdt_predictions)
    print(cm_sdt)

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Evaluating Random Forest...")
    rf_predictions = rf_model.predict(X_test)
    print("RF Classification Report:")
    print(classification_report(y_test, rf_predictions))
    print("RF Confusion Matrix:")
    cm_rf = confusion_matrix(y_test, rf_predictions)
    print(cm_rf)

    # Train SNN+RF Hybrid
    print("Training SNN+RF Hybrid...")
    snn = NeuralFeatureExtractor(input_dim=X_train.shape[1], output_dim=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(snn.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = snn(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    X_train_transformed = snn(X_train_tensor).detach().numpy()
    rf_hybrid = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf_hybrid.fit(X_train_transformed, y_train)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_transformed = snn(X_test_tensor).detach().numpy()
    snn_rf_predictions = rf_hybrid.predict(X_test_transformed)

    print("SNN+RF Classification Report:")
    print(classification_report(y_test, snn_rf_predictions))
    print("SNN+RF Confusion Matrix:")
    cm_snn_rf = confusion_matrix(y_test, snn_rf_predictions)
    print(cm_snn_rf)

    # Save models
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(sdt, "sdt_model.pkl")
    joblib.dump(rf_hybrid, "snn_rf_model.pkl")

    # Plot Confusion Matrices
    for cm, title in zip([cm_sdt, cm_rf, cm_snn_rf], ["SDT", "RF", "SNN+RF"]):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.title(f"{title} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"{title.lower()}_confusion_matrix.png")
        plt.show()

# Run the script
if __name__ == "__main__":
    train_and_evaluate()
