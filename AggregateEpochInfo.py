import re
import matplotlib.pyplot as plt

# Replace this with your actual log data (or read from a file)
log_data = """
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-001/eeg/sub-001_task-sitstand_eeg.set
Epochs Created: 40 epochs.
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-002/eeg/sub-002_task-sitstand_eeg.set
Epochs Created: 36 epochs.
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-003/eeg/sub-003_task-sitstand_eeg.set
Epochs Created: 37 epochs.
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-004/eeg/sub-004_task-sitstand_eeg.set
Epochs Created: 37 epochs.
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-005/eeg/sub-005_task-sitstand_eeg.set
Epochs Created: 34 epochs.
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-006/eeg/sub-006_task-sitstand_eeg.set
Epochs Created: 33 epochs.
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-007/eeg/sub-007_task-sitstand_eeg.set
Epochs Created: 35 epochs.
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-008/eeg/sub-008_task-sitstand_eeg.set
Epochs Created: 36 epochs.
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-009/eeg/sub-009_task-sitstand_eeg.set
Epochs Created: 40 epochs.
EEG Data Loaded: /Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data/sub-010/eeg/sub-010_task-sitstand_eeg.set
Epochs Created: 34 epochs.
"""

# Extract participant IDs and epoch counts
participants = []
epoch_counts = []

for line in log_data.splitlines():
    if "EEG Data Loaded" in line:
        participant_match = re.search(r"sub-\d+", line)
        if participant_match:
            participants.append(participant_match.group(0))
    elif "Epochs Created" in line:
        epochs_match = re.search(r"Epochs Created: (\d+)", line)
        if epochs_match:
            epoch_counts.append(int(epochs_match.group(1)))

# Ensure data integrity
if len(participants) != len(epoch_counts):
    print("Warning: Mismatch between participants and epoch counts. Please check log data.")
else:
    # Sort participants and epoch counts by participant ID
    sorted_data = sorted(zip(participants, epoch_counts), key=lambda x: x[0])
    participants, epoch_counts = zip(*sorted_data)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(participants, epoch_counts, color='skyblue')
    plt.xticks(rotation=45, ha="right")
    plt.title("Epoch Retention per Participant", fontsize=16)
    plt.xlabel("Participant", fontsize=14)
    plt.ylabel("Retained Epochs", fontsize=14)
    plt.tight_layout()
    plt.savefig("epoch_retention_plot.png")  # Save plot as a PNG file
    plt.show()
