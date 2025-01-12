
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import torch
from joblib import load
from model import SpikingDecisionTree, NeuralFeatureExtractor


# Seed for reproducibility
np.random.seed(42)


def preprocess_data(filepath, feature_dim, smote=False):
    """Preprocess network dataset."""
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)

    # Handle categorical and missing data
    non_numeric_columns = data.select_dtypes(include=["object"]).columns
    if len(non_numeric_columns) > 0:
        print(f"Handling non-numeric columns: {non_numeric_columns.tolist()}")
        for col in non_numeric_columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["label"])  # Drop rows where the target column is missing
    y = data["label"]
    X = data.drop(columns=["label"])
    
    print("Handling missing values...")
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    if smote:
        print("Balancing dataset using SMOTE...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print("Balanced class distribution:", np.unique(y, return_counts=True))

    # Align feature dimensions
    if X.shape[1] < feature_dim:
        print(f"Adding {feature_dim - X.shape[1]} missing columns")
        padding = np.zeros((X.shape[0], feature_dim - X.shape[1]))
        X = np.hstack([X, padding])
    elif X.shape[1] > feature_dim:
        print(f"Truncating features to {feature_dim} columns.")
        X = X[:, :feature_dim]

    return X.astype(np.float32), y


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    print(f"\nEvaluating {model_name}...")
    print("\nSample Predictions:", np.unique(predictions, return_counts=True))
    print("\nSample True Labels:", np.unique(y_test, return_counts=True))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"\n{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1-score: {f1:.4f}")

    return accuracy, precision, recall, f1


def test_network_dataset():
    """Main function to test models on network dataset."""
    base_path = "/Users/shanezhu/Desktop/Research/USENIX ATC '25"
    network_dataset = f"{base_path}/train_test_network.csv"
    feature_dim = 8561  # Match model training dimension
    pca_components = 32  # Match PCA components for SNN+RF

    print("Loading models...")
    try:
        # Load models
        sdt_model = load(f"{base_path}/Model/sdt_model.pkl")
        rf_model = load(f"{base_path}/Model/rf_model.pkl")
        rf_hybrid = load(f"{base_path}/Model/snn_rf_model.pkl")
        snn = NeuralFeatureExtractor(input_dim=feature_dim, output_dim=64)
        snn.load_state_dict(torch.load(f"{base_path}/Model/snn_feature_extractor.pth", map_location=torch.device("cpu")))
        snn.eval()
        pca = load(f"{base_path}/Model/pca.pkl")
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    try:
        # Preprocess without SMOTE for SDTs
        print("\nProcessing dataset without SMOTE for SDTs...")
        X_test_sdt, y_test_sdt = preprocess_data(network_dataset, feature_dim, smote=False)
        
        # Preprocess with SMOTE for RF and SNN+RF
        print("\nProcessing dataset with SMOTE for RF and SNN+RF...")
        X_test_rf, y_test_rf = preprocess_data(network_dataset, feature_dim, smote=True)

        # Test Spiking Decision Tree
        print("\nEvaluating Spiking Decision Tree...")
        evaluate_model(sdt_model, X_test_sdt, y_test_sdt, "Spiking Decision Tree")

        # Test Random Forest
        print("\nEvaluating Random Forest...")
        evaluate_model(rf_model, X_test_rf, y_test_rf, "Random Forest")

        # Inspect feature importances for RF
        if hasattr(rf_model, "feature_importances_"):
            print("\nRandom Forest Feature Importances (Top 10):")
            print(sorted(enumerate(rf_model.feature_importances_), key=lambda x: -x[1])[:10])

        # Transform features using SNN
        print("\nTransforming features using SNN for SNN+RF Hybrid...")
        X_test_tensor = torch.tensor(X_test_rf, dtype=torch.float32)
        with torch.no_grad():
            X_transformed = snn(X_test_tensor).numpy()

        print("\nApplying PCA...")
        X_test_pca = pca.transform(X_transformed)

        # Test SNN+RF Hybrid
        print("\nEvaluating SNN+RF Hybrid...")
        evaluate_model(rf_hybrid, X_test_pca, y_test_rf, "SNN+RF Hybrid")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_network_dataset()
