import numpy as np
import os

processed_spike_trains_path = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Processed Spike Trains"

# List all spike train files
spike_train_files = [f for f in os.listdir(processed_spike_trains_path) if f.endswith('.npy')]

# Dictionary to store summary statistics
summary_stats = {}

for file in spike_train_files:
    try:
        # Load the spike train data
        spike_trains = np.load(os.path.join(processed_spike_trains_path, file), allow_pickle=True)
        spike_counts = [np.sum(epoch) for epoch in spike_trains]
        
        # Calculate summary statistics
        avg_spike_count = np.mean(spike_counts)
        spike_variance = np.var(spike_counts)
        
        # Store statistics in dictionary
        summary_stats[file] = {
            "Average Spike Count": avg_spike_count,
            "Variance": spike_variance,
            "Number of Epochs": len(spike_trains),
        }
    except Exception as e:
        print(f"Error processing file {file}: {e}")

for file, stats in summary_stats.items():
    print(f"File: {file}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
