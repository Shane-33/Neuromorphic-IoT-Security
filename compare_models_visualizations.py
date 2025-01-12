import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def compare_models_visualizations(sdt_metrics, rf_metrics, snn_rf_metrics, feature_importance_rf, feature_importance_snn):
    """
    Generates visualizations comparing SDT, RF, and SNN+RF.

    Arguments:
    - sdt_metrics: Dictionary containing SDT metrics.
    - rf_metrics: Dictionary containing RF metrics.
    - snn_rf_metrics: Dictionary containing SNN+RF metrics.
    - feature_importance_rf: Feature importance values for RF.
    - feature_importance_snn: Feature importance values for SNN+RF.
    """

    # 1. Confusion Matrix Comparison
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    sns.heatmap(sdt_metrics['conf_matrix'], annot=True, fmt="d", cmap="Blues", ax=axs[0])
    axs[0].set_title("SDT Confusion Matrix")
    sns.heatmap(rf_metrics['conf_matrix'], annot=True, fmt="d", cmap="Blues", ax=axs[1])
    axs[1].set_title("RF Confusion Matrix")
    sns.heatmap(snn_rf_metrics['conf_matrix'], annot=True, fmt="d", cmap="Blues", ax=axs[2])
    axs[2].set_title("SNN+RF Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_comparison.png")
    plt.show()

    # 2. Performance Metrics Bar Chart
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    sdt_values = [sdt_metrics[m] for m in metrics]
    rf_values = [rf_metrics[m] for m in metrics]
    snn_rf_values = [snn_rf_metrics[m] for m in metrics]
    x = np.arange(len(metrics))

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.25, sdt_values, width=0.25, label="SDT")
    plt.bar(x, rf_values, width=0.25, label="RF")
    plt.bar(x + 0.25, snn_rf_values, width=0.25, label="SNN+RF")
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Performance Metrics Comparison")
    plt.legend()
    plt.savefig("performance_metrics_comparison.png")
    plt.show()

    # 3. ROC Curve Comparison
    plt.figure(figsize=(10, 6))
    for model, metrics in [("SDT", sdt_metrics), ("RF", rf_metrics), ("SNN+RF", snn_rf_metrics)]:
        fpr, tpr = metrics["roc_curve"]
        auc_value = metrics["auc"]
        plt.plot(fpr, tpr, label=f"{model} (AUC = {auc_value:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve_comparison.png")
    plt.show()

    # 4. Feature Importance Comparison
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance_rf)), feature_importance_rf, color='blue', label="RF")
    plt.barh(range(len(feature_importance_snn)), feature_importance_snn, color='green', alpha=0.6, label="SNN+RF")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance Comparison")
    plt.legend()
    plt.savefig("feature_importance_comparison.png")
    plt.show()

    # 5. Learning Curve Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sdt_metrics['epochs'], sdt_metrics['train_accuracies'], label="SDT - Training")
    plt.plot(sdt_metrics['epochs'], sdt_metrics['val_accuracies'], label="SDT - Validation")
    plt.plot(rf_metrics['epochs'], rf_metrics['train_accuracies'], label="RF - Training")
    plt.plot(rf_metrics['epochs'], rf_metrics['val_accuracies'], label="RF - Validation")
    plt.plot(snn_rf_metrics['epochs'], snn_rf_metrics['train_accuracies'], label="SNN+RF - Training")
    plt.plot(snn_rf_metrics['epochs'], snn_rf_metrics['val_accuracies'], label="SNN+RF - Validation")
    plt.title("Learning Curve Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve_comparison.png")
    plt.show()

    # 6. Scatter Plot Comparing Metrics
    plt.figure(figsize=(10, 6))
    models = ["SDT", "RF", "SNN+RF"]
    accuracies = [sdt_metrics["Accuracy"], rf_metrics["Accuracy"], snn_rf_metrics["Accuracy"]]
    recalls = [sdt_metrics["Recall"], rf_metrics["Recall"], snn_rf_metrics["Recall"]]
    plt.scatter(models, accuracies, label="Accuracy", color='blue', s=100)
    plt.scatter(models, recalls, label="Recall", color='green', s=100)
    plt.title("Scatter Plot: Accuracy vs Recall")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("scatter_plot_metrics.png")
    plt.show()


# Example Metrics (Replace with actual results)
sdt_metrics = {
    "conf_matrix": np.array([[29400, 594], [600, 29394]]),
    "Accuracy": 0.99,
    "Precision": 0.98,
    "Recall": 0.99,
    "F1-Score": 0.99,
    "roc_curve": ([0.0, 0.1, 0.2], [0.0, 0.8, 1.0]),
    "auc": 0.95,
    "epochs": range(1, 11),
    "train_accuracies": np.random.uniform(0.9, 0.99, 10),
    "val_accuracies": np.random.uniform(0.88, 0.98, 10)
}

rf_metrics = {
    "conf_matrix": np.array([[29994, 0], [0, 29994]]),
    "Accuracy": 1.00,
    "Precision": 1.00,
    "Recall": 1.00,
    "F1-Score": 1.00,
    "roc_curve": ([0.0, 0.1, 0.2], [0.0, 0.9, 1.0]),
    "auc": 0.98,
    "epochs": range(1, 11),
    "train_accuracies": np.random.uniform(0.99, 1.0, 10),
    "val_accuracies": np.random.uniform(0.98, 0.99, 10)
}

snn_rf_metrics = {
    "conf_matrix": np.array([[0, 29994], [0, 29994]]),
    "Accuracy": 0.50,
    "Precision": 0.25,
    "Recall": 0.50,
    "F1-Score": 0.33,
    "roc_curve": ([0.0, 0.1, 0.2], [0.0, 0.5, 1.0]),
    "auc": 0.50,
    "epochs": range(1, 11),
    "train_accuracies": np.random.uniform(0.5, 0.6, 10),
    "val_accuracies": np.random.uniform(0.45, 0.55, 10)
}

feature_importance_rf = [0.2, 0.3, 0.1, 0.25, 0.15]
feature_importance_snn = [0.18, 0.28, 0.12, 0.22, 0.2]

# Generate Visualizations
compare_models_visualizations(sdt_metrics, rf_metrics, snn_rf_metrics, feature_importance_rf, feature_importance_snn)
