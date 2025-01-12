import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Paths
SPIKE_TRAIN_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Processed Spike Trains"

# Load spike train files
def load_spike_trains(directory):
    spike_train_files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith("_spike_trains.npy")
    ]
    spike_trains = {}
    for file in spike_train_files:
        try:
            participant_id = os.path.basename(file).split("_")[0]  # Extract participant ID
            spike_trains[participant_id] = np.load(file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    print(f"Loaded spike trains for {len(spike_trains)} participants.")
    return spike_trains

# Perform clustering on spike train patterns
def cluster_spike_patterns(spike_trains, n_clusters=3):
    """
    Cluster spike train patterns using KMeans.
    """
    # Flatten spike trains for clustering
    flattened_trains = []
    labels = []
    for participant_id, spikes in spike_trains.items():
        for epoch_idx, epoch_spikes in enumerate(spikes):
            flattened_trains.append(epoch_spikes.flatten())
            labels.append((participant_id, epoch_idx))
    
    flattened_trains = np.array(flattened_trains)
    print(f"Clustering {len(flattened_trains)} spike trains...")
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(flattened_trains)
    print("Clustering complete.")
    
    return cluster_labels, labels, kmeans

# Visualize cluster centers
def visualize_clusters(kmeans, n_clusters=3):
    """
    Visualize cluster centers as representative spike train patterns.
    """
    cluster_centers = kmeans.cluster_centers_
    time_points = cluster_centers.shape[1]

    plt.figure(figsize=(10, 6))
    for i in range(n_clusters):
        plt.plot(
            range(time_points),
            cluster_centers[i],
            label=f"Cluster {i + 1}",
        )
    plt.title("Cluster Centers for Spike Train Patterns")
    plt.xlabel("Time Points")
    plt.ylabel("Spike Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Step 1: Load spike trains
    spike_trains = load_spike_trains(SPIKE_TRAIN_DIR)

    # Step 2: Perform clustering
    n_clusters = 3  # Adjust the number of clusters as needed
    cluster_labels, labels, kmeans = cluster_spike_patterns(spike_trains, n_clusters)

    # Step 3: Visualize cluster centers
    visualize_clusters(kmeans, n_clusters)
