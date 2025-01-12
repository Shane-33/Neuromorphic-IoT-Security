import numpy as np
import torch
import torch.nn as nn

class SpikingDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return {"label": np.argmax(np.bincount(y))}

        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return {"label": np.argmax(np.bincount(y))}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node = {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

        return node

    def _information_gain(self, parent, left_child, right_child):
        def entropy(y):
            proportions = np.bincount(y) / len(y)
            return -np.sum([p * np.log2(p) for p in proportions if p > 0])

        n = len(parent)
        n_l, n_r = len(left_child), len(right_child)
        return entropy(parent) - (n_l / n * entropy(left_child) + n_r / n * entropy(right_child))

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, node):
        if "label" in node:
            return node["label"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_single(x, node["left"])
        else:
            return self._predict_single(x, node["right"])


class NeuralFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Optional: Add activation function for spiking behavior
class SpikingActivation(nn.Module):
    def __init__(self, threshold=1.0):
        super(SpikingActivation, self).__init__()
        self.threshold = threshold
        
    def forward(self, x):
        return torch.where(x > self.threshold, torch.ones_like(x), torch.zeros_like(x))