def validate_balanced_data(output_dir):
    for file in os.listdir(output_dir):
        if file.endswith("_spike_trains.npy"):
            balanced_data = np.load(os.path.join(output_dir, file))
            
            # Count benign and malicious samples
            labels = balanced_data[:, -1]
            benign_count = np.sum(labels == 0)
            malicious_count = np.sum(labels != 0)
            
            print(f"{file}: Benign={benign_count}, Malicious={malicious_count}")

if __name__ == "__main__":
    print("Validating balanced datasets...")
    validate_balanced_data(BALANCED_OUTPUT_DIR)
