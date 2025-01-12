import os
import pandas as pd

# Input and Output Directories
IOT_DATASET_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/IoTScenarios"
OUTPUT_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/ProcessedFeatures"

# Column Dtypes
dtype_dict = {
    "col_6": str, "col_7": float, "col_8": float, "col_9": float,
    "col_10": int, "col_11": int, "col_12": str
}

# Chunk Size for Processing
CHUNK_SIZE = 50000  # Adjust as necessary for memory optimization

def extract_features_from_log(file_path, output_file_name):
    """
    Extracts features from a single IoT-23 dataset log file and saves to a CSV.
    """
    try:
        # Process file in chunks
        selected_features_list = []
        for chunk in pd.read_csv(
            file_path,
            sep='\s+',  # Handle whitespace-separated data
            skiprows=8,  # Skip metadata rows
            chunksize=CHUNK_SIZE,  # Process in chunks to save memory
            dtype=dtype_dict,  # Specify dtypes for mixed columns
            low_memory=False
        ):
            # Dynamically rename columns to avoid duplicates
            chunk.columns = [f"col_{i}" for i in range(chunk.shape[1])]

            # Select relevant columns
            relevant_columns = [
                "col_6",  # Protocol
                "col_7",  # Duration
                "col_8",  # Origin Bytes
                "col_9",  # Response Bytes
                "col_10", # Origin Packets
                "col_11", # Response Packets
                "col_12"  # Label
            ]
            selected_features = chunk[relevant_columns]

            # Rename columns for clarity
            selected_features.columns = [
                "protocol", "duration", "orig_bytes", "resp_bytes", 
                "orig_pkts", "resp_pkts", "label"
            ]
            selected_features_list.append(selected_features)

        # Concatenate all chunks and save to CSV
        final_features = pd.concat(selected_features_list, ignore_index=True)
        final_features.to_csv(output_file_name, index=False)
        print(f"Features saved to {output_file_name}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_all_files(input_dir, output_dir):
    """
    Processes all `.log.labeled` files in the IoT-23 dataset directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".log.labeled"):
                input_file_path = os.path.join(root, file)

                # Generate a unique output file name
                relative_path = os.path.relpath(root, input_dir)  # Get directory structure
                unique_name = os.path.join(relative_path, file).replace("/", "_")
                output_file_path = os.path.join(output_dir, f"{unique_name}_features.csv")

                # Process the file
                extract_features_from_log(input_file_path, output_file_path)

if __name__ == "__main__":
    print("Extracting features from IoT-23 dataset with optimized memory usage...")
    process_all_files(IOT_DATASET_DIR, OUTPUT_DIR)
