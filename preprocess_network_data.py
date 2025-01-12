import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
DATA_FOLDER = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Processed_Network_dataset/"
OUTPUT_FILE = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Processed_Network_dataset/preprocessed_network_combined.csv"

# Visualization function
def visualize_data(data, title, stage):
    """Visualize dataset distribution at various preprocessing stages."""
    print(f"Visualizing {stage}...")
    
    plt.figure(figsize=(12, 8))
    
    # Visualize missing data
    missing = data.isnull().sum()
    sns.barplot(x=missing.index, y=missing.values)
    plt.title(f"Missing Data After {stage}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualize numeric feature distributions
    numeric_cols = data.select_dtypes(include=["number"]).columns
    if not numeric_cols.empty:
        print(f"Visualizing numeric data distribution after {stage}...")
        data[numeric_cols].hist(bins=20, figsize=(14, 10))
        plt.suptitle(f"Numeric Feature Distributions After {stage}", fontsize=16)
        plt.tight_layout()
        plt.show()

# Preprocessing the dataset
def preprocess_network_datasets(data_folder, output_file):
    # List all files in the folder
    print("Listing dataset files...")
    all_files = [f for f in os.listdir(data_folder) if f.startswith("Network_dataset_") and f.endswith(".csv")]
    print(f"Found {len(all_files)} dataset files.")

    combined_data = pd.DataFrame()

    for file in all_files:
        file_path = os.path.join(data_folder, file)
        print(f"Processing {file_path}...")
        data = pd.read_csv(file_path, low_memory=False)

        # Step 1: Visualize raw data
        visualize_data(data, title="Raw Data", stage="Initial Load")

        # Step 2: Handle missing values
        print("  Handling missing values...")
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Visualize missing values after handling
        visualize_data(data, title="After Handling Missing Values", stage="Missing Values")

        # Step 3: Ensure numeric columns are clean
        numeric_columns = [
            'src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes',
            'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes'
        ]
        for column in numeric_columns:
            if column in data.columns:
                print(f"  Cleaning numeric column: {column}")
                data[column] = pd.to_numeric(data[column], errors='coerce')  # Convert to float, invalid entries become NaN

        # Step 4: Handle any new missing values caused by invalid entries
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Visualize numeric data cleaning
        visualize_data(data, title="After Numeric Data Cleaning", stage="Numeric Cleaning")

        # Step 5: Encode categorical features
        categorical_columns = ['src_ip', 'dst_ip', 'proto', 'service', 'conn_state', 'type']
        label_encoders = {}
        for column in categorical_columns:
            if column in data.columns:
                print(f"  Encoding categorical column: {column}")
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                label_encoders[column] = le

        # Visualize after encoding
        visualize_data(data, title="After Encoding Categorical Features", stage="Encoding")

        # Step 6: Scale numeric features
        print("  Scaling numeric features...")
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # Visualize scaling effects
        visualize_data(data, title="After Scaling Numeric Features", stage="Scaling")

        # Append processed data to combined dataset
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Step 7: Save combined preprocessed dataset
    print(f"Saving combined preprocessed dataset to {output_file}...")
    combined_data.to_csv(output_file, index=False)
    print("Preprocessing complete. Combined dataset saved.")

# Run preprocessing
if __name__ == "__main__":
    preprocess_network_datasets(DATA_FOLDER, OUTPUT_FILE)