import pandas as pd
import os

# Path to EEG data
eeg_data_path = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data"

# Iterate through all event files
for root, dirs, files in os.walk(eeg_data_path):
    for file in files:
        if file.endswith("_events.tsv"):
            file_path = os.path.join(root, file)
            
            try:
                # Load and sort events
                events = pd.read_csv(file_path, sep='\t')
                events = events.sort_values(by='onset')
                events.to_csv(file_path, sep='\t', index=False)
                print(f"Sorted and saved: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
