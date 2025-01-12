import os
import numpy as np
import pandas as pd

# Directories
PROCESSED_FEATURES_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/ProcessedFeatures"
SPIKE_OUTPUT_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/SpikeTrains"

CHUNK_SIZE = 10000  # Adjust this value based on your system's memory capacity

def encode_spikes(features_chunk, threshold=500, coding="rate"):
    """
    Encodes features into spike trains using the specified coding strategy for a chunk of data.
    """
    spike_trains = []
    for _, row in features_chunk.iterrows():
        if coding == "rate":
            spike_row = (row[["orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]] > threshold).astype(int)
        elif coding == "temporal":
            spike_row = row[["orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]] // threshold
            spike_row = spike_row.apply(lambda x: min(x, 10))  # Cap to avoid excessive spikes
        spike_trains.append(spike_row.values)
    return np.array(spike_trains)

def process_features_chunked(input_dir, output_dir, threshold=500, coding="rate"):
    """
    Processes feature files in chunks, encodes them as spike trains, and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith("_features.csv"):
            print(f"Processing {file}...")
            input_file_path = os.path.join(input_dir, file)
            output_file_path = os.path.join(output_dir, f"{file}_spike_trains.npy")

            # Initialize an empty list to accumulate spike trains
            all_spike_trains = []

            # Process the file in chunks
            for chunk in pd.read_csv(input_file_path, chunksize=CHUNK_SIZE, low_memory=False):
                # Convert relevant columns to numeric, coercing invalid values to NaN
                for col in ["orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

                # Handle missing values by replacing NaN with column medians
                for col in ["orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]:
                    if chunk[col].isnull().any():
                        median_value = chunk[col].median()
                        chunk[col].fillna(median_value, inplace=True)

                # Encode the chunk into spike trains
                spike_trains_chunk = encode_spikes(chunk, threshold=threshold, coding=coding)
                all_spike_trains.append(spike_trains_chunk)

            # Concatenate all spike trains and save to a .npy file
            final_spike_trains = np.concatenate(all_spike_trains, axis=0)
            np.save(output_file_path, final_spike_trains)
            print(f"Spike trains saved to {output_file_path}")

if __name__ == "__main__":
    print("Encoding IoT-23 features into spike trains...")
    process_features_chunked(PROCESSED_FEATURES_DIR, SPIKE_OUTPUT_DIR, threshold=500, coding="rate")
