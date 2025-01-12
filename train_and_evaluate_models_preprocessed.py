import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from norse.torch.functional.lif import LIFParameters, LIFState, lif_feed_forward_step

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

# SNN Feature Extractor
class SNNFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNNFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif_params = LIFParameters()

    def forward(self, x):
        lif_state1 = LIFState(
            v=torch.zeros(x.size(0), self.fc1.out_features, dtype=x.dtype, device=x.device),
            i=torch.zeros(x.size(0), self.fc1.out_features, dtype=x.dtype, device=x.device),
            z=torch.zeros(x.size(0), self.fc1.out_features, dtype=x.dtype, device=x.device)
        )
        z1 = self.fc1(x)
        s1, lif_state1 = lif_feed_forward_step(z1, lif_state1, self.lif_params)

        lif_state2 = LIFState(
            v=torch.zeros(x.size(0), self.fc2.out_features, dtype=x.dtype, device=x.device),
            i=torch.zeros(x.size(0), self.fc2.out_features, dtype=x.dtype, device=x.device),
            z=torch.zeros(x.size(0), self.fc2.out_features, dtype=x.dtype, device=x.device)
        )
        z2 = self.fc2(s1)
        s2, lif_state2 = lif_feed_forward_step(z2, lif_state2, self.lif_params)

        return s2

# Load and preprocess data
def load_and_preprocess():
    print("Loading preprocessed data...")
    X_combined = np.load("X_combined.npy")
    y = np.load("y.npy")

    print("Handling missing values...")
    imputer = SimpleImputer(strategy="mean")
    X_combined = imputer.fit_transform(X_combined)

    print("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_combined, y = smote.fit_resample(X_combined, y)
    print(f"Resampled Features Shape: {X_combined.shape}, Resampled Labels Shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42, stratify=y)
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
    print(classification_report(y_test, sdt_predictions, zero_division=0))
    cm_sdt = confusion_matrix(y_test, sdt_predictions)

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Evaluating Random Forest...")
    rf_predictions = rf_model.predict(X_test)
    print("RF Classification Report:")
    print(classification_report(y_test, rf_predictions, zero_division=0))
    cm_rf = confusion_matrix(y_test, rf_predictions)

    # Train SNN+RF Hybrid
    print("Training SNN+RF Hybrid...")
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 64
    snn = SNNFeatureExtractor(input_size, hidden_size, output_size)
    snn.eval()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        snn_features_train = snn(X_train_tensor).cpu().numpy()
        snn_features_test = snn(X_test_tensor).cpu().numpy()

    print("Scaling and Applying PCA...")
    scaler = StandardScaler()
    snn_features_train = scaler.fit_transform(snn_features_train)
    snn_features_test = scaler.transform(snn_features_test)

    pca = PCA(n_components=min(output_size, 32))
    snn_features_train = pca.fit_transform(snn_features_train)
    snn_features_test = pca.transform(snn_features_test)

    rf_hybrid = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf_hybrid.fit(snn_features_train, y_train)

    snn_rf_predictions = rf_hybrid.predict(snn_features_test)
    print("SNN+RF Classification Report:")
    print(classification_report(y_test, snn_rf_predictions, zero_division=0))
    cm_snn_rf = confusion_matrix(y_test, snn_rf_predictions)

    # Save models, scaler, and PCA
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(sdt, "sdt_model.pkl")
    joblib.dump(rf_hybrid, "snn_rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca.pkl")

    # Save the feature extractor
    torch.save(snn.state_dict(), "snn_feature_extractor.pth")

    # Save feature dimension to a file
    with open("feature_dimension.txt", "w") as f:
        f.write(str(input_size))

    print("Models, scaler, PCA, and feature dimension saved successfully.")


    # Plot Confusion Matrices
    for cm, title in zip([cm_sdt, cm_rf, cm_snn_rf], ["SDT", "RF", "SNN+RF"]):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f"{title} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"{title.lower()}_confusion_matrix.png")
        plt.show()

# Run the script
if __name__ == "__main__":
    train_and_evaluate()