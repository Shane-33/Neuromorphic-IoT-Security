import mne
import numpy as np
import pandas as pd
import os

# Constants
BANDPASS_FREQS = (0.5, 40)  # Band-pass filter range in Hz
EPOCH_START = -0.5  # Start of epoch relative to event (in seconds)
EPOCH_END = 1.5  # End of epoch relative to event (in seconds)
SPIKE_THRESHOLD = 0.5  # Threshold for spike encoding (rate coding example)

# Paths
BASE_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/EEG Data"
OUTPUT_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/Processed Spike Trains"

# Preprocessing and spike encoding functions
def load_eeg(file_path):
    """Load EEG data from a .set file."""
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    print(f"EEG Data Loaded: {file_path}")
    return raw

def preprocess_eeg(raw):
    """Apply band-pass filter to EEG data."""
    raw.filter(BANDPASS_FREQS[0], BANDPASS_FREQS[1], fir_design='firwin')
    print("EEG data preprocessed with band-pass filtering.")
    return raw

def load_events(event_file_path):
    """Load events from a .tsv file."""
    try:
        events_df = pd.read_csv(event_file_path, sep='\t')
        print(f"Loaded event file: {event_file_path}")
        
        # Ensure numeric values and remove commas
        for col in ['onset', 'duration', 'sample', 'value']:
            if col in events_df.columns:
                events_df[col] = pd.to_numeric(
                    events_df[col].replace(',', '', regex=True), errors='coerce'
                )
        
        # Drop invalid rows
        events_df.dropna(subset=['onset', 'value'], inplace=True)

        # Convert to MNE-compatible events array
        events = []
        for _, row in events_df.iterrows():
            events.append([int(row['onset']), 0, int(row['value'])])
        events = np.array(events)

        print(f"Events Loaded: {len(events)} events found.")
        return events
    except Exception as e:
        print(f"Error loading events from {event_file_path}: {e}")
        return None

def segment_data(raw, events, event_id):
    """
    Create epochs for a specific event ID.
    Parameters:
        raw: MNE Raw object containing EEG data.
        events: Numpy array of event markers.
        event_id: Integer ID of the event to segment data for.
    Returns:
        epochs: MNE Epochs object for the specified event.
    """
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=EPOCH_START, tmax=EPOCH_END,
        baseline=(None, 0), preload=True
    )
    print(f"Epochs Created: {len(epochs)} epochs.")
    return epochs


def spike_encoding(epochs, threshold=SPIKE_THRESHOLD):
    """
    Encode epochs into spike trains based on a threshold.
    Parameters:
        epochs: MNE Epochs object containing segmented EEG data.
        threshold: Float, threshold for spike encoding (0-1 normalized range).
    Returns:
        spike_trains: List of binary spike trains (1 for spike, 0 otherwise).
    """
    spike_trains = []
    for epoch_idx, epoch in enumerate(epochs.get_data()):
        # Debug: Check the range of epoch data before thresholding
        epoch_min = np.min(epoch)
        epoch_max = np.max(epoch)
        epoch_mean = np.mean(epoch)
        print(f"Epoch {epoch_idx} Stats - Min: {epoch_min}, Max: {epoch_max}, Mean: {epoch_mean}")

        # Ensure data has sufficient range to normalize (avoid division by zero)
        if epoch_max > epoch_min:
            # Normalize epoch data to 0-1 range
            normalized_epoch = (epoch - epoch_min) / (epoch_max - epoch_min)
        else:
            # If all values are the same, create a flat zero array
            normalized_epoch = np.zeros_like(epoch)
            print(f"Epoch {epoch_idx} has no variation and was set to zero.")

        # Debug: Check normalized data range
        print(f"Normalized Epoch {epoch_idx} Stats - Min: {np.min(normalized_epoch)}, Max: {np.max(normalized_epoch)}")

        # Thresholding for spikes
        spikes = (normalized_epoch > threshold).astype(int)
        spike_trains.append(spikes)

    print(f"Spike trains generated for {len(spike_trains)} epochs.")

    # Debug: Check overall spike train statistics
    all_spike_values = np.concatenate(spike_trains).flatten()
    print(f"Spike Train Stats - Min: {np.min(all_spike_values)}, Max: {np.max(all_spike_values)}, Mean: {np.mean(all_spike_values)}")

    return spike_trains


def save_spike_trains(spike_trains, participant_id):
    """Save spike trains as a .npy file."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_file = os.path.join(OUTPUT_DIR, f"sub-{participant_id:03d}_spike_trains.npy")
    np.save(output_file, spike_trains)
    print(f"Spike trains saved for participant sub-{participant_id:03d} at {output_file}")

# Main workflow
def process_all_participants():
    """Process EEG data for all participants."""
    for participant_id in range(1, 33):  # Adjust range based on available participants
        participant_folder = f"sub-{participant_id:03d}"
        eeg_file_path = os.path.join(BASE_DIR, participant_folder, "eeg", f"{participant_folder}_task-sitstand_eeg.set")
        event_file_path = os.path.join(BASE_DIR, participant_folder, "eeg", f"{participant_folder}_task-sitstand_events.tsv")
        
        # Check if files exist
        if not os.path.exists(eeg_file_path) or not os.path.exists(event_file_path):
            print(f"Files missing for {participant_folder}. Skipping...")
            continue
        
        try:
            # Step 1: Load and preprocess EEG data
            raw = load_eeg(eeg_file_path)
            raw = preprocess_eeg(raw)
            
            # Step 2: Load and process events
            events = load_events(event_file_path)
            if events is None or len(events) == 0:
                print(f"Skipping {participant_folder} due to event loading issue.")
                continue
            
            # Define event IDs for specific tasks
            event_id = {'MotorImageryA': 1, 'IdleStateA': 2, 'MotorImageryB': 3, 'IdleStateB': 4}
            
            # Step 3: Segment data into epochs for MotorImageryA task
            epochs_motor_imagery = segment_data(raw, events, event_id['MotorImageryA'])
            
            # Step 4: Convert to spike trains
            spike_trains = spike_encoding(epochs_motor_imagery)
            
            # Step 5: Save spike trains
            save_spike_trains(spike_trains, participant_id)
        
        except Exception as e:
            print(f"Error processing {participant_folder}: {e}")

if __name__ == "__main__":
    process_all_participants()
