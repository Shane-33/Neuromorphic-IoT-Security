import pandas as pd
import os

# Paths
input_file = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset_subset.csv"
output_file = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset_2GB.csv"

# Approximate number of rows for 2 GB (based on average row size)
# Adjust the number of rows as needed to match 2 GB.
target_size_gb = 2
row_size_kb = 25 / 1_000_000  # Calculate row size in KB
rows_for_2GB = int((target_size_gb * 1_000_000) / row_size_kb)

print(f"Target size: {target_size_gb} GB, Estimated rows: {rows_for_2GB}")

# Load the input dataset in chunks and sample rows
chunk_size = 100_000
sampled_data = []

print("Sampling rows...")
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    sampled_data.append(chunk.sample(frac=0.1, random_state=42))  # Adjust sampling fraction as needed
    if len(pd.concat(sampled_data)) >= rows_for_2GB:
        break

# Concatenate sampled chunks
subset_data = pd.concat(sampled_data).head(rows_for_2GB)

# Save the subset dataset
print(f"Saving the sampled dataset to {output_file}...")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
subset_data.to_csv(output_file, index=False)

print("Subset dataset created successfully.")
