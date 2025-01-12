import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load preprocessed data
print("Loading preprocessed data...")
X_combined = np.load("X_combined.npy")
y = np.load("y.npy")
print(f"Data loaded. Features Shape: {X_combined.shape}, Labels Shape: {y.shape}")

# Handle missing values
print("Handling missing values...")
imputer = SimpleImputer(strategy="mean")
X_combined = imputer.fit_transform(X_combined)

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE to the training set only
print("Applying SMOTE to the training set...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train the Random Forest model
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,  # Reduced for regularization
    max_depth=15,      # Limit depth to avoid overfitting
    random_state=42
)
rf_model.fit(X_train, y_train)

# Evaluate the model on the test set
print("Evaluating on the test set...")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Cross-validation to validate generalization
print("Performing cross-validation...")
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Random Forest (Post-SMOTE)")
plt.savefig("confusion_matrix_rf_post_smote.png")
plt.show()

# Print final accuracy
print(f"Test Set Accuracy: {accuracy}")
