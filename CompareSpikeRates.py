import os
import numpy as np
import matplotlib.pyplot as plt

processed_spike_trains_dir = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Processed Spike Trains"

spike_train_files = [
    os.path.join(processed_spike_trains_dir, file)
    for file in os.listdir(processed_spike_trains_dir)
    if file.endswith("_spike_trains.npy")
]

if not spike_train_files:
    raise FileNotFoundError("No spike train files found in the specified directory.")

spike_counts_per_file = []

# Process each spike train file
for spike_train_file in spike_train_files:
    try:
        spike_trains = np.load(spike_train_file, allow_pickle=True)
        total_spike_counts = np.sum(spike_trains)  # Total spikes across all epochs
        avg_spike_count_per_epoch = np.mean(np.sum(spike_trains, axis=1))  # Avg spikes per epoch
        spike_counts_per_file.append(avg_spike_count_per_epoch)

        print(f"{spike_train_file}: Total Spikes: {total_spike_counts}, Avg Spikes Per Epoch: {avg_spike_count_per_epoch:.2f}")
    except Exception as e:
        print(f"Error processing file {spike_train_file}: {e}")

participants = [os.path.basename(file).split("_")[0] for file in spike_train_files]

plt.figure(figsize=(10, 6))
plt.bar(participants, spike_counts_per_file, color='skyblue', edgecolor='black')
plt.xlabel("Participants")
plt.ylabel("Average Spike Count per Epoch")
plt.title("Average Spike Count Per Epoch for Each Participant")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
