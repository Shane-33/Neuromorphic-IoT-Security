import numpy as np  # Import numpy for array handling
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
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
        # Initialize state dynamically based on input size
        self.lif_state = LIFState(
            v=torch.zeros(x.size(0), self.fc1.out_features, dtype=x.dtype, device=x.device),
            i=torch.zeros(x.size(0), self.fc1.out_features, dtype=x.dtype, device=x.device),
            z=torch.zeros(x.size(0), self.fc1.out_features, dtype=x.dtype, device=x.device)
        )

        # First layer
        z1 = self.fc1(x)
        s1, self.lif_state = lif_feed_forward_step(z1, self.lif_state, self.lif_params)

        # Second layer
        self.lif_state = LIFState(
            v=torch.zeros(x.size(0), self.fc2.out_features, dtype=x.dtype, device=x.device),
            i=torch.zeros(x.size(0), self.fc2.out_features, dtype=x.dtype, device=x.device),
            z=torch.zeros(x.size(0), self.fc2.out_features, dtype=x.dtype, device=x.device)
        )
        z2 = self.fc2(s1)
        s2, self.lif_state = lif_feed_forward_step(z2, self.lif_state, self.lif_params)

        return s2

# Load preprocessed data
X_combined = np.load("X_combined.npy")
y = np.load("y.npy")

# Convert to PyTorch tensors
X_combined_tensor = torch.tensor(X_combined, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Initialize SNN
input_size = X_combined.shape[1]
hidden_size = 64
output_size = 32  # Reduced dimensionality for RF
snn = SNNFeatureExtractor(input_size, hidden_size, output_size)

# Extract features using SNN
print("Extracting features with SNN...")
snn.eval()
with torch.no_grad():
    snn_features_train = snn(X_train_tensor).cpu().numpy()
    snn_features_test = snn(X_test_tensor).cpu().numpy()

# Train Random Forest Classifier
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(snn_features_train, y_train)

# Save the trained model
joblib.dump(rf_model, "rf_model.pkl")

# Evaluate Random Forest
print("Evaluating Random Forest...")
y_pred = rf_model.predict(snn_features_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SNN + Random Forest")
plt.savefig("confusion_matrix_snn_rf.png")
plt.show()
