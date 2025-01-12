import os
import numpy as np

SPIKE_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/SpikeTrains"
BALANCED_OUTPUT_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/BalancedSpikeTrains"

def balance_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith("_spike_trains.npy"):
            spike_trains = np.load(os.path.join(input_dir, file))
            
            # Split into benign and malicious
            labels = spike_trains[:, -1]  # Assuming the last column is the label
            benign = spike_trains[labels == 0]
            malicious = spike_trains[labels != 0]
            
            # Oversample malicious to match benign count
            if len(benign) > len(malicious):
                malicious = np.tile(malicious, (len(benign) // len(malicious) + 1, 1))[:len(benign)]
            elif len(malicious) > len(benign):
                benign = np.tile(benign, (len(malicious) // len(benign) + 1, 1))[:len(malicious)]
            
            balanced_data = np.concatenate([benign, malicious])
            np.random.shuffle(balanced_data)  # Shuffle to mix classes
            
            output_file_path = os.path.join(output_dir, file)
            np.save(output_file_path, balanced_data)
            print(f"Balanced dataset saved to {output_file_path}")

if __name__ == "__main__":
    print("Balancing spike train dataset...")
    balance_dataset(SPIKE_DIR, BALANCED_OUTPUT_DIR)
