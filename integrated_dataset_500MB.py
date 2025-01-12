import pandas as pd
import os

# Paths
input_file = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset_subset.csv"
output_file = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Integrated_Data/integrated_dataset_500MB.csv"

# Total size of the dataset in MB
total_file_size_mb = 25_890  # Approx 25.89 GB
total_rows = 1_000_000  # Example: replace with actual row count if known

# Calculate average row size
average_row_size_kb = (total_file_size_mb * 1_000) / total_rows  # Average row size in KB
target_size_mb = 500  # Target size (adjust for 500 MB or other sizes)
rows_for_target_size = int((target_size_mb * 1_000) / average_row_size_kb)

print(f"Dataset Size: {total_file_size_mb} MB")
print(f"Estimated average row size: {average_row_size_kb:.2f} KB")
print(f"Rows for {target_size_mb} MB: {rows_for_target_size}")

# Load the input dataset in chunks and sample rows
chunk_size = 100_000
sampled_data = []

print("Sampling rows...")
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    sampled_data.append(chunk.sample(frac=0.01, random_state=42))  # Adjust fraction based on row count
    if len(pd.concat(sampled_data)) >= rows_for_target_size:
        break

# Concatenate sampled chunks
subset_data = pd.concat(sampled_data).head(rows_for_target_size)

# Save the subset dataset
print(f"Saving the sampled dataset to {output_file}...")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
subset_data.to_csv(output_file, index=False)

print(f"Subset dataset of ~{target_size_mb} MB created successfully.")
