import pandas as pd
import ast  # Use ast.literal_eval for safe string evaluation
import matplotlib.pyplot as plt

# Path to integrated data
INTEGRATED_DATA_PATH = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset_500MB.csv"

# Load and analyze integrated data
def analyze_integrated_data(data_path):
    # Load the integrated dataset in chunks for efficiency
    print("Loading integrated data...")
    chunksize = 100_000
    chunk_list = []
    for chunk in pd.read_csv(data_path, chunksize=chunksize):
        chunk_list.append(chunk)
    data = pd.concat(chunk_list, ignore_index=True)
    print(f"Integrated data loaded with shape: {data.shape}")

    # Check the structure of the data
    print("Data head:")
    print(data.head())

    # Basic statistics
    print("Basic statistics:")
    print(data.describe())

    # Correlation analysis
    numeric_data = data.select_dtypes(include=['number'])
    print("Correlation analysis:")
    correlation = numeric_data.corr()
    print(correlation)

    # Efficiently process spike_train column if it exists
    if 'spike_train' in data.columns:
        print("Processing and visualizing IoT feature vs spike train activity...")

        def compute_spike_sum(spike_str):
            try:
                # Use ast.literal_eval for safe string parsing
                nested_list = ast.literal_eval(spike_str)
                return sum(sum(sublist) for sublist in nested_list)
            except (ValueError, SyntaxError):
                return 0  # Handle invalid entries gracefully

        # Apply processing with progress reporting
        print("Computing spike train sums...")
        data['spike_train_sum'] = data['spike_train'].map(compute_spike_sum)

        # Scatter plot: IoT feature vs spike train activity
        if 'src_pkts' in data.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(data['src_pkts'], data['spike_train_sum'], alpha=0.5)
            plt.xlabel("Source Packets")
            plt.ylabel("Spike Train Activity")
            plt.title("Source Packets vs EEG Spike Train Activity")
            plt.show()
        else:
            print("Source Packets ('src_pkts') column not found.")
    else:
        print("'spike_train' column not found in the data.")

# Run analysis
if __name__ == "__main__":
    analyze_integrated_data(INTEGRATED_DATA_PATH)
