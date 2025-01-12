import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from norse.torch.functional.lif import LIFParameters, LIFState, lif_feed_forward_step
import joblib

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
print("Loading preprocessed data...")
X_combined = np.load("X_combined.npy")
y = np.load("y.npy")
print(f"Data loaded. Features Shape: {X_combined.shape}, Labels Shape: {y.shape}")

# Handle missing values
print("Handling missing values...")
imputer = SimpleImputer(strategy="mean")
X_combined = imputer.fit_transform(X_combined)

# Handle class imbalance
print("Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_combined, y = smote.fit_resample(X_combined, y)
print(f"Resampled Features Shape: {X_combined.shape}, Resampled Labels Shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42, stratify=y)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Initialize SNN
input_size = X_combined.shape[1]
hidden_size = 128

# Tuning for output_size and Random Forest hyperparameters
output_sizes = [32, 64, 128]
rf_hyperparams = [{"n_estimators": 100, "max_depth": 20}, {"n_estimators": 200, "max_depth": 15}]
best_accuracy = 0
best_model = None
best_params = {}
best_cm = None

train_accuracies = []
val_accuracies = []

for output_size in output_sizes:
    for params in rf_hyperparams:
        print(f"Tuning SNN output size: {output_size}, RF params: {params}")
        snn = SNNFeatureExtractor(input_size, hidden_size, output_size)
        snn.eval()
        with torch.no_grad():
            snn_features_train = snn(X_train_tensor).cpu().numpy()
            snn_features_test = snn(X_test_tensor).cpu().numpy()

        # Apply PCA
        print("Applying PCA...")
        try:
            pca = PCA(n_components=min(output_size, 32))
            snn_features_train = pca.fit_transform(snn_features_train)
            snn_features_test = pca.transform(snn_features_test)
        except Exception as e:
            print(f"PCA failed: {e}")
            continue

        # Visualize PCA explained variance
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.savefig(f'pca_explained_variance_{output_size}.png')
        plt.show()

        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)
        rf_model.fit(snn_features_train, y_train)

        # Evaluate
        y_pred = rf_model.predict(snn_features_test)
        accuracy = accuracy_score(y_test, y_pred)
        train_accuracies.append(rf_model.score(snn_features_train, y_train))
        val_accuracies.append(accuracy)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = rf_model
            best_params = {"output_size": output_size, "n_estimators": params["n_estimators"], "max_depth": params["max_depth"]}
            best_cm = cm

# Visualize learning curves
plt.figure()
plt.plot(train_accuracies, label="Training Accuracy", marker="o")
plt.plot(val_accuracies, label="Validation Accuracy", marker="o")
plt.xlabel("Model Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Learning Curve for SNN+RF")
plt.savefig("snn_rf_learning_curve.png")
plt.show()

print(f"Best Model Accuracy: {best_accuracy}, Params: {best_params}")

# Save the best model
joblib.dump(best_model, "best_rf_model.pkl")

# Plot the best confusion matrix
if best_cm is not None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for Best SNN + Random Forest")
    plt.savefig("best_confusion_matrix_snn_rf.png")
    plt.show()
