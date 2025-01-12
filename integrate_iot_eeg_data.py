import pandas as pd
import numpy as np
import os

# Paths
IOT_DATA_PATH = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Processed_Network_dataset/preprocessed_network_combined.csv"
EEG_SPIKE_TRAINS_FOLDER = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Processed Spike Trains/"
OUTPUT_PATH = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset_subset.csv"

def integrate_iot_eeg_subset(iot_data_path, eeg_folder, output_path, max_rows=1_000_000, iot_chunk_size=100_000):
    """
    Integrate IoT and EEG data efficiently and save a subset in CSV format.
    """
    try:
        # Check if EEG folder exists
        if not os.path.exists(eeg_folder):
            print(f"Error: EEG spike train folder '{eeg_folder}' does not exist.")
            return

        # Load EEG spike train files
        eeg_files = [f for f in os.listdir(eeg_folder) if f.endswith("_spike_trains.npy")]
        if not eeg_files:
            print("No EEG spike train files found.")
            return
        print(f"Found {len(eeg_files)} EEG spike train files.")

        # Open CSV writer to write rows incrementally
        subset_written = 0
        header_written = False

        print("Processing IoT data in chunks...")
        # Process IoT data in chunks to reduce memory usage
        for iot_chunk in pd.read_csv(iot_data_path, chunksize=iot_chunk_size, low_memory=False):
            print(f"Processing IoT chunk with shape: {iot_chunk.shape}")
            iot_chunk['ts'] = pd.to_datetime(iot_chunk['ts'], errors='coerce')
            iot_chunk = iot_chunk.dropna(subset=['ts']).sort_values(by='ts').reset_index(drop=True)

            for eeg_file in eeg_files:
                participant_id = eeg_file.split("_")[0]
                eeg_path = os.path.join(eeg_folder, eeg_file)
                print(f"Processing EEG data for participant: {participant_id}")

                # Load EEG spike trains
                spike_trains = np.load(eeg_path, allow_pickle=True)
                num_epochs = len(spike_trains)

                # Downsample EEG timestamps (adjust interval as needed, e.g., 10s)
                eeg_timestamps = pd.date_range(
                    start=iot_chunk['ts'].iloc[0], periods=num_epochs, freq="10s"
                )

                # Create a temporary DataFrame for EEG timestamps and spike trains
                eeg_df = pd.DataFrame({
                    'ts': eeg_timestamps,
                    'spike_train': [spike.tolist() for spike in spike_trains]
                })

                # Merge IoT chunk with EEG data
                merged_data = pd.merge_asof(
                    iot_chunk, eeg_df, on='ts', direction='nearest', tolerance=pd.Timedelta("1s")
                )

                # Randomly sample rows from the merged data
                sampled_data = merged_data.sample(
                    n=min(max_rows - subset_written, len(merged_data)), random_state=42
                )

                # Append sampled rows to the CSV file incrementally
                if not sampled_data.empty:
                    sampled_data.to_csv(output_path, mode='a', header=not header_written, index=False)
                    header_written = True  # Ensure header is written only once
                    subset_written += len(sampled_data)

                print(f"Subset written so far: {subset_written} rows.")

                # Stop if the maximum number of rows has been written
                if subset_written >= max_rows:
                    print(f"Reached the target subset size: {max_rows} rows.")
                    return

        print(f"Subset integration complete. Total rows written: {subset_written}.")
    except Exception as e:
        print(f"An error occurred during integration: {e}")

# Run the integration
if __name__ == "__main__":
    integrate_iot_eeg_subset(IOT_DATA_PATH, EEG_SPIKE_TRAINS_FOLDER, OUTPUT_PATH)
