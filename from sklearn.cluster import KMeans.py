from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Combine all spike trains into a single matrix for clustering
all_spike_patterns = []

for file in spike_train_files:
    spike_trains = np.load(os.path.join(processed_spike_trains_path, file))
    for epoch in spike_trains:
        # Convert spike times to a fixed-length vector (e.g., 50 ms bins)
        spike_histogram, _ = np.histogram(epoch, bins=np.linspace(-0.5, 0.5, 51))
        all_spike_patterns.append(spike_histogram)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(all_spike_patterns)

# Visualize cluster centers
for i, center in enumerate(kmeans.cluster_centers_):
    plt.plot(center, label=f"Cluster {i}")
plt.legend()
plt.xlabel("Time Bin")
plt.ylabel("Spike Count")
plt.title("Clustered Spike Train Patterns")
plt.show()
