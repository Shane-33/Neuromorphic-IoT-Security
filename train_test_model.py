import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_PATH = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Train_Test_Network_dataset/train_test_network.csv"

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {data.shape}")

    # Separate features and labels
    X = data.drop(columns=['label', 'type'], errors='ignore')  # Drop target columns
    y = data['label']  # Binary target (0 = Normal, 1 = Attack)

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
    return X, y

# Train and evaluate the model
def train_and_evaluate_model(X, y):
    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train a Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Quick Benchmarking')
    plt.savefig("confusion_matrix_quick_benchmark.png")
    plt.show()

    print("Accuracy:", accuracy_score(y_test, y_pred))

# Main function
if __name__ == "__main__":
    X, y = load_and_preprocess_data(DATA_PATH)
    train_and_evaluate_model(X, y)
