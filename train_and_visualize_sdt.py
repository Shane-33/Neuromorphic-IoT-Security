import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns


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
            "threshold": np.mean(X[:, 0]) if len(X) > 0 else 0,  # Split based on mean
            "feature": 0  # Simple feature selection (can extend to multiple features)
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


# Train and Visualize Spiking Decision Tree
def train_and_visualize_sdt():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Train SDT
    print("Training Spiking Decision Tree...")
    sdt = SpikingDecisionTree(max_depth=10)

    train_accuracies = []
    val_accuracies = []

    for depth in range(1, 11):  # Test different depths
        sdt.max_depth = depth
        sdt.fit(X_train, y_train)

        # Evaluate on training data
        train_preds = sdt.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        train_accuracies.append(train_acc)

        # Evaluate on validation data
        val_preds = sdt.predict(X_test)
        val_acc = accuracy_score(y_test, val_preds)
        val_accuracies.append(val_acc)

        print(f"Depth {depth} - Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Plot Learning Curve with Generalization Gap
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), train_accuracies, label="Training Accuracy", marker="o")
    plt.plot(range(1, 11), val_accuracies, label="Validation Accuracy", marker="o")
    plt.fill_between(range(1, 11), train_accuracies, val_accuracies, color='gray', alpha=0.2, label="Generalization Gap")
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve for Spiking Decision Tree")
    plt.legend()
    plt.grid()
    plt.savefig("sdt_learning_curve_with_gap.png")
    plt.show()


    # Evaluate Final SDT
    print("Evaluating Final SDT...")
    sdt.max_depth = 10
    sdt.fit(X_train, y_train)
    final_preds = sdt.predict(X_test)
    print("SDT Classification Report:")
    print(classification_report(y_test, final_preds))
    print("SDT Confusion Matrix:")
    cm = confusion_matrix(y_test, final_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title("Confusion Matrix for Spiking Decision Tree")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("sdt_confusion_matrix.png")
    plt.show()



    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Example inputs (Replace these with your actual outputs)
    train_accuracies = [0.82, 0.91, 0.96, 0.975, 0.977, 0.977, 0.977, 0.977, 0.977, 0.977]
    val_accuracies = [0.825, 0.93, 0.965, 0.975, 0.977, 0.977, 0.977, 0.977, 0.977, 0.977]
    depths = range(1, 11)  # Tree depths
    cm = np.array([[30085, 0], [750, 29153]])  # Example confusion matrix
    feature_importance = [0.3, 0.25, 0.2, 0.15, 0.1]  # Example feature importances
    features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]

    # 1. Learning Curve with Generalization Gap
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accuracies, label="Training Accuracy", marker="o", linestyle="--")
    plt.plot(depths, val_accuracies, label="Validation Accuracy", marker="o", linestyle="-")
    plt.fill_between(depths, train_accuracies, val_accuracies, color='gray', alpha=0.2, label="Generalization Gap")
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve for Spiking Decision Tree")
    plt.legend()
    plt.grid()
    plt.savefig("sdt_learning_curve.png")
    plt.show()

    # 2. Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=features, palette="viridis")
    plt.title("Feature Importance in Spiking Decision Tree")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.savefig("sdt_feature_importance.png")
    plt.show()

    # 3. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Malware"], yticklabels=["Normal", "Malware"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for Spiking Decision Tree")
    plt.savefig("sdt_confusion_matrix.png")
    plt.show()

    # 4. Training vs Validation Accuracy as Separate Plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Training Accuracy
    axs[0].plot(depths, train_accuracies, label="Training Accuracy", color="blue", marker="o")
    axs[0].set_title("Training Accuracy Across Depths")
    axs[0].set_xlabel("Tree Depth")
    axs[0].set_ylabel("Accuracy")
    axs[0].grid()

    # Validation Accuracy
    axs[1].plot(depths, val_accuracies, label="Validation Accuracy", color="green", marker="o")
    axs[1].set_title("Validation Accuracy Across Depths")
    axs[1].set_xlabel("Tree Depth")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid()

    plt.tight_layout()
    plt.savefig("sdt_training_vs_validation.png")
    plt.show()

    # 5. Overlayed Feature Distributions (Optional, Replace X and y with your dataset)
    Example: X is feature data, y is labels
    sns.histplot(data=X[y == 0, 0], label="Normal", color="blue", kde=True, alpha=0.6)
    sns.histplot(data=X[y == 1, 0], label="Malware", color="red", kde=True, alpha=0.6)
    plt.title("Feature 1 Distribution by Class")
    plt.xlabel("Feature 1 Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.savefig("sdt_feature_distribution.png")
    plt.show()



if __name__ == "__main__":
    train_and_visualize_sdt()
