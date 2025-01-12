from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from joblib import load
from sklearn.decomposition import PCA
from model import SpikingDecisionTree, NeuralFeatureExtractor


def preprocess_data(filepath, feature_dim, smote_strategy=None):
    """Preprocess network dataset with optional SMOTE strategy."""
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)

    # Handle non-numeric columns
    non_numeric_columns = data.select_dtypes(include=["object"]).columns
    if len(non_numeric_columns) > 0:
        print(f"Handling non-numeric columns: {non_numeric_columns.tolist()}")
        for col in non_numeric_columns:
            if col == "label":
                continue
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows where the target column is missing
    data = data.dropna(subset=["label"])
    y = data["label"]
    X = data.drop(columns=["label"])

    # Handle missing values
    print("Handling missing values...")
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Align feature dimensions
    if X.shape[1] < feature_dim:
        print(f"Aligning features: Adding {feature_dim - X.shape[1]} missing columns")
        missing_features = np.zeros((X.shape[0], feature_dim - X.shape[1]))
        X = np.hstack([X, missing_features])
    elif X.shape[1] > feature_dim:
        print(f"Truncating features: Reducing to {feature_dim} columns.")
        X = X[:, :feature_dim]

    # Apply SMOTE if specified
    if smote_strategy:
        print(f"Applying {smote_strategy.__class__.__name__}...")
        X, y = smote_strategy.fit_resample(X, y)
        print(f"Balanced class distribution: {np.unique(y, return_counts=True)}")

    return X.astype(np.float32), y.values


def test_network_dataset():
    """Main function to test models on network dataset."""
    base_path = "/Users/shanezhu/Desktop/Research/USENIX ATC '25"
    network_dataset = f"{base_path}/train_test_network.csv"
    feature_dim = 8561
    pca_components = 32

    print("Loading models...")
    try:
        sdt_model = load(f"{base_path}/sdt_model.pkl")
        rf_model = load(f"{base_path}/rf_model.pkl")
        rf_hybrid = load(f"{base_path}/snn_rf_model.pkl")
        snn = NeuralFeatureExtractor(input_dim=feature_dim, output_dim=64)
        snn.load_state_dict(torch.load(f"{base_path}/Model/snn_feature_extractor.pth", map_location=torch.device("cpu")))
        snn.eval()
        pca = load(f"{base_path}/pca.pkl")
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    try:
        print("\nProcessing dataset without SMOTE for SDTs...")
        X_test_sdt, y_test_sdt = preprocess_data(network_dataset, feature_dim)

        print("\nProcessing dataset with SMOTE for RF...")
        rf_smote = SMOTE(sampling_strategy=0.8)
        X_test_rf, y_test_rf = preprocess_data(network_dataset, feature_dim, smote_strategy=rf_smote)

        print("\nProcessing dataset with Borderline-SMOTE for SNN+RF...")
        snn_rf_smote = BorderlineSMOTE(sampling_strategy=0.6)
        X_test_snn_rf, y_test_snn_rf = preprocess_data(network_dataset, feature_dim, smote_strategy=snn_rf_smote)

        # Evaluate Spiking Decision Tree
        print("\nEvaluating Spiking Decision Tree...")
        evaluate_model(sdt_model, X_test_sdt, y_test_sdt, "Spiking Decision Tree")

        # Evaluate Random Forest
        print("\nEvaluating Random Forest...")
        evaluate_model(rf_model, X_test_rf, y_test_rf, "Random Forest")

        # Transform features using SNN and Evaluate SNN+RF
        print("\nTransforming features using SNN for SNN+RF...")
        X_test_snn_rf_tensor = torch.tensor(X_test_snn_rf, dtype=torch.float32)
        with torch.no_grad():
            X_test_snn_rf_transformed = snn(X_test_snn_rf_tensor).numpy()

        print("\nApplying PCA...")
        X_test_snn_rf_pca = pca.transform(X_test_snn_rf_transformed)

        print("\nEvaluating SNN+RF Hybrid...")
        evaluate_model(rf_hybrid, X_test_snn_rf_pca, y_test_snn_rf, "SNN+RF Hybrid")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    print(f"\nEvaluating {model_name}...")
    print("\nSample Predictions:", np.unique(predictions, return_counts=True))
    print("\nSample True Labels:", np.unique(y_test, return_counts=True))

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print(f"\n{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1-score: {f1:.4f}")


if __name__ == "__main__":
    test_network_dataset()
