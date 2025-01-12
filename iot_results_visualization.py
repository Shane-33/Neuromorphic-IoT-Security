import matplotlib.pyplot as plt
import numpy as np

# Models and datasets
models = ["Spiking Decision Tree", "Random Forest", "SNN+RF Hybrid"]
datasets = ["Fridge", "Garage Door", "GPS Tracker", "Modbus", "Thermostat"]

# Updated metrics
accuracy_data = {
    "Spiking Decision Tree": [62.45, 61.90, 59.80, 51.00, 53.00],
    "Random Forest": [64.20, 62.80, 61.50, 53.50, 56.10],
    "SNN+RF Hybrid": [65.00, 64.10, 62.00, 54.00, 57.50]
}

precision_data = {
    "Spiking Decision Tree": [30, 29, 28, 26, 25],
    "Random Forest": [35, 34, 33, 31, 30],
    "SNN+RF Hybrid": [38, 37, 36, 34, 33]
}

recall_data = {
    "Spiking Decision Tree": [90, 89, 88, 85, 83],
    "Random Forest": [92, 91, 90, 87, 85],
    "SNN+RF Hybrid": [94, 93, 92, 89, 87]
}

# Visualization 1: Accuracy across datasets
plt.figure(figsize=(10, 6))
for model in models:
    plt.plot(datasets, accuracy_data[model], marker='o', label=model)
plt.title("Model Accuracy on IoT Datasets", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("IoT Datasets", fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig("model_accuracy_iot_datasets.png")
plt.show()

# Visualization 2: Precision across datasets
plt.figure(figsize=(10, 6))
for model in models:
    plt.plot(datasets, precision_data[model], marker='s', label=model)
plt.title("Model Precision on IoT Datasets", fontsize=14)
plt.ylabel("Precision (%)", fontsize=12)
plt.xlabel("IoT Datasets", fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig("model_precision_iot_datasets.png")
plt.show()

# Visualization 3: Recall across datasets
plt.figure(figsize=(10, 6))
for model in models:
    plt.plot(datasets, recall_data[model], marker='^', label=model)
plt.title("Model Recall on IoT Datasets", fontsize=14)
plt.ylabel("Recall (%)", fontsize=12)
plt.xlabel("IoT Datasets", fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig("model_recall_iot_datasets.png")
plt.show()

# Visualization 4: Accuracy Comparison (Bar Chart)
bar_width = 0.25
x = np.arange(len(datasets))

plt.figure(figsize=(12, 6))
for i, model in enumerate(models):
    plt.bar(x + i * bar_width, accuracy_data[model], width=bar_width, label=model)

plt.title("Model Accuracy Comparison (IoT Datasets)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("IoT Datasets", fontsize=12)
plt.xticks(x + bar_width, datasets)
plt.legend()
plt.grid(axis='y')
plt.savefig("model_accuracy_comparison_bar.png")
plt.show()

# Visualization 5: Precision vs Recall
plt.figure(figsize=(10, 6))
for model in models:
    plt.plot(datasets, precision_data[model], marker='o', label=f"{model} Precision", linestyle='--')
    plt.plot(datasets, recall_data[model], marker='x', label=f"{model} Recall", linestyle='-')
plt.title("Precision vs Recall on IoT Datasets", fontsize=14)
plt.ylabel("Metric (%)", fontsize=12)
plt.xlabel("IoT Datasets", fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig("precision_vs_recall_iot_datasets.png")
plt.show()
