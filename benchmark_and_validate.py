import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
TRAIN_TEST_DATA_PATH = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Train_Test_Network_dataset/train_test_network.csv"
INTEGRATED_DATA_PATH = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset.csv"

# Load and preprocess data
def load_and_preprocess_data(file_path):
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {data.shape}")

    # Check label distribution
    print("Label Distribution:")
    print(data['label'].value_counts())

    # Separate features and labels
    X = data.drop(columns=['label', 'type', 'spike_train'], errors='ignore')
    y = data['label']

    # Encode categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"Encoding categorical column: {col}")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("Preprocessing complete.")
    return X, y, scaler

# Train and evaluate the model
def train_and_evaluate_model(X, y, integrated_data_path=None):
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    print("Validation Set Classification Report:")
    y_val_pred = model.predict(X_val)
    print(classification_report(y_val, y_val_pred))

    # Evaluate on test set
    print("Test Set Classification Report:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix for test set
    cm = confusion_matrix(y_test, y_test_pre
