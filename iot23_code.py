import pandas as pd

def load_iot23_data(data_dir):
    """Loads the IoT-23 dataset.

    Args:
        data_dir: Path to the directory containing the IoT-23 dataset.

    Returns:
        A pandas DataFrame containing the network traffic data.
    """
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".log.labeled"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath, sep=" ")
            data.append(df)
    return pd.concat(data, ignore_index=True)

# Example usage
data_dir = "/path/to/iot23_dataset"
df = load_iot23_data(data_dir)


# Handle missing values
df = df.fillna(method='ffill') # Forward fill missing values

# Remove duplicate rows
df = df.drop_duplicates()


df['packet_size'] = df['resp_pkts'] * df['resp_bytes'] + df['orig_pkts'] * df['orig_bytes']
df['flow_duration'] = df['duration']




