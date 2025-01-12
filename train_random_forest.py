import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and preprocess data
print("Loading data...")
X_combined = np.load("X_combined.npy")
y = np.load("y.npy")
print(f"Data loaded. Features Shape: {X_combined.shape}, Labels Shape: {y.shape}")

# Handle missing values
print("Handling missing values...")
X_combined = np.nan_to_num(X_combined)

# Handle class imbalance
print("Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_combined, y = smote.fit_resample(X_combined, y)
print(f"Resampled Features Shape: {X_combined.shape}, Resampled Labels Shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42, stratify=y)

# Apply PCA
print("Applying PCA...")
pca = PCA(n_components=50)  # Adjust n_components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train Random Forest without PCA
print("Training Random Forest without PCA...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest without PCA
print("Evaluating Random Forest without PCA...")
y_pred_no_pca = rf_model.predict(X_test)
accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)
print(f"Accuracy without PCA: {accuracy_no_pca}")
print("Classification Report without PCA:")
print(classification_report(y_test, y_pred_no_pca))
print("Confusion Matrix without PCA:")
cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
sns.heatmap(cm_no_pca, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix for Random Forest without PCA")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Train Random Forest with PCA
print("Training Random Forest with PCA...")
rf_model_pca = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_model_pca.fit(X_train_pca, y_train)

# Evaluate Random Forest with PCA
print("Evaluating Random Forest with PCA...")
y_pred_pca = rf_model_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"Accuracy with PCA: {accuracy_pca}")
print("Classification Report with PCA:")
print(classification_report(y_test, y_pred_pca))
print("Confusion Matrix with PCA:")
cm_pca = confusion_matrix(y_test, y_pred_pca)
sns.heatmap(cm_pca, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix for Random Forest with PCA")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save models
joblib.dump(rf_model, "rf_model_no_pca.pkl")
joblib.dump(rf_model_pca, "rf_model_with_pca.pkl")
