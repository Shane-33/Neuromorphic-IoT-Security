import pandas as pd
import os

# Paths
input_file = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset_subset.csv"
output_file = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset_50MB.csv"

# Load a sample to calculate row size
data_sample = pd.read_csv(input_file, nrows=1000)
average_row_size_bytes = data_sample.memory_usage(deep=True).sum() / len(data_sample)

# Target size in bytes for 50 MB
target_size_bytes = 50 * 1_000_000
rows_for_50MB = int(target_size_bytes / average_row_size_bytes)
print(f"Target file size: 50 MB, Estimated rows: {rows_for_50MB}")

# Load the input dataset in chunks and sample rows
chunk_size = 100_000
sampled_data = []

print("Sampling rows...")
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    sampled_data.append(chunk.sample(frac=0.01, random_state=42))  # Adjust sampling fraction as needed
    if len(pd.concat(sampled_data)) >= rows_for_50MB:
        break

# Concatenate sampled chunks
subset_data = pd.concat(sampled_data).head(rows_for_50MB)

# Save the subset dataset
print(f"Saving the sampled dataset to {output_file}...")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
subset_data.to_csv(output_file, index=False)

print("Subset dataset created successfully.")
