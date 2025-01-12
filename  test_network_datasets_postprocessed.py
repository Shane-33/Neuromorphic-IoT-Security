import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import torch
from joblib import load
from model import NeuralFeatureExtractor


class SpikingDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples == 0 or depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_indices], y[left_indices], depth + 1),
            "right": self._build_tree(X[right_indices], y[right_indices], depth + 1),
        }

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        n = len(y)
        left_weight = np.sum(left_indices) / n
        right_weight = np.sum(right_indices) / n

        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _entropy(self, y):
        hist = np.bincount(y, minlength=2)
        ps = hist / np.sum(hist)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, dict):
            if inputs[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node


def preprocess_data(filepath, feature_dim):
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)

    non_numeric_columns = data.select_dtypes(include=["object"]).columns
    for col in non_numeric_columns:
        if col == "label":
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["label"])
    y = data["label"].astype(int)
    X = data.drop(columns=["label"])

    print("Handling missing values...")
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    if X.shape[1] < feature_dim:
        print(f"Adding {feature_dim - X.shape[1]} missing columns")
        X = np.hstack([X, np.zeros((X.shape[0], feature_dim - X.shape[1]))])
    elif X.shape[1] > feature_dim:
        print(f"Reducing features to {feature_dim} dimensions.")
        X = X[:, :feature_dim]

    return X.astype(np.float32), y.values


def evaluate_model_cv(model, X, y, model_name, cv_splits=3):
    print(f"\nEvaluating {model_name} with {cv_splits}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        })

    avg_results = {k: np.mean([r[k] for r in results]) for k in results[0]}
    print(f"Results: {avg_results}")
    return avg_results


def test_network_dataset():
    base_path = "/Users/shanezhu/Desktop/Research/USENIX ATC '25"
    dataset_path = f"{base_path}/train_test_network.csv"
    feature_dim = 8561

    print("Loading models...")
    sdt_model = SpikingDecisionTree(max_depth=5, min_samples_split=10)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    pca = load(f"{base_path}/Model/pca.pkl")

    X, y = preprocess_data(dataset_path, feature_dim)

    evaluate_model_cv(sdt_model, X, y, "Spiking Decision Tree")
    evaluate_model_cv(rf_model, X, y, "Random Forest")


if __name__ == "__main__":
    test_network_dataset()


