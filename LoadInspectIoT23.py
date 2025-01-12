import os

IOT_DATASET_DIR = "/Users/shanezhu/Desktop/Research/USENIX ATC '25/IoTScenarios"

def inspect_dataset(directory):
    for root, dirs, files in os.walk(directory):
        print(f"Directory: {root}")
        for file in files:
            print(f"  File: {file}")

if __name__ == "__main__":
    print("Inspecting IoT-23 dataset structure...")
    inspect_dataset(IOT_DATASET_DIR)
