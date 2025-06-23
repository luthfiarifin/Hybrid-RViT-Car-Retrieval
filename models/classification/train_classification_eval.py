import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)


class DatasetExplorer:
    """
    A class for exploring and visualizing the dataset for multiple directories (e.g., train and val).
    """

    def __init__(self, train_dir, val_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.train_class_counts = self.explore_dataset(train_dir)
        self.val_class_counts = self.explore_dataset(val_dir)
        self.class_names = sorted(
            set(self.train_class_counts.keys()) | set(self.val_class_counts.keys())
        )

    @staticmethod
    def explore_dataset(data_dir):
        """Explore the dataset structure and class distribution."""
        import os

        if not os.path.exists(data_dir):
            print(f"Dataset directory {data_dir} not found!")
            return {}

        class_counts = {}
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                count = len(
                    [
                        f
                        for f in os.listdir(class_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                class_counts[class_name] = count
        return class_counts

    def summary_report(self):
        """Print and visualize the dataset class distribution and summary."""
        import matplotlib.pyplot as plt
        import pandas as pd

        # Visualize class distribution
        classes = self.class_names
        train_values = [self.train_class_counts.get(cls, 0) for cls in classes]
        val_values = [self.val_class_counts.get(cls, 0) for cls in classes]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.bar(classes, train_values, color="skyblue", alpha=0.7)
        ax1.set_title(
            "Training Set - Class Distribution", fontsize=14, fontweight="bold"
        )
        ax1.set_xlabel("Car Types")
        ax1.set_ylabel("Number of Images")
        ax1.tick_params(axis="x", rotation=45)
        for i, v in enumerate(train_values):
            ax1.text(
                i,
                v + max(train_values) * 0.01 if train_values else 0,
                str(v),
                ha="center",
                fontweight="bold",
            )
        ax2.bar(classes, val_values, color="lightcoral", alpha=0.7)
        ax2.set_title(
            "Validation Set - Class Distribution", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Car Types")
        ax2.set_ylabel("Number of Images")
        ax2.tick_params(axis="x", rotation=45)
        for i, v in enumerate(val_values):
            ax2.text(
                i,
                v + max(val_values) * 0.01 if val_values else 0,
                str(v),
                ha="center",
                fontweight="bold",
            )
        plt.tight_layout()
        plt.show()


class TrainingEvaluation:
    """
    Generates comprehensive training analysis, visualizations, and reports for car classification model training.
    """

    def __init__(self, trainer, config, training_results):
        self.trainer = trainer
        self.config = config
        self.results = training_results
        self.class_names = trainer.train_dataset.classes
        self.class_counts = self._get_class_counts()

        # Set style for plots
        plt.style.use("default")
        sns.set_palette("husl")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def _get_class_counts(self):
        class_counts = np.zeros(len(self.class_names))
        for _, target in self.trainer.train_dataset:
            class_counts[target] += 1
        return class_counts

    def plot_training_progress(self):
        """Visualize training/validation loss, accuracy, learning rate, and summary stats."""
        epochs = range(1, len(self.results["train_losses"]) + 1)
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))

        # Loss curves
        ax1.plot(
            epochs,
            self.results["train_losses"],
            "b-",
            label="Training Loss",
            linewidth=2,
        )
        ax1.plot(
            epochs,
            self.results["val_losses"],
            "r-",
            label="Validation Loss",
            linewidth=2,
        )
        ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curve
        ax2.plot(
            epochs,
            self.results["val_accuracies"],
            "g-",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax2.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])

        # Training time per epoch
        ax3.bar(epochs, self.results["epoch_times"], color="orange", alpha=0.7)
        ax3.set_title("Training Time per Epoch", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Time (seconds)")
        ax3.grid(True, alpha=0.3)

        # Learning rate schedule
        ax4.plot(epochs, self.results["learning_rates"], "purple", linewidth=2)
        ax4.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate")
        ax4.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax4.grid(True, alpha=0.3)

        # Early Stopping Counter History
        if "early_stopping_counter_history" in self.results:
            ax5.plot(
                epochs,
                self.results["early_stopping_counter_history"],
                "m-",
                linewidth=2,
            )
            ax5.set_title(
                "Early Stopping Counter History", fontsize=14, fontweight="bold"
            )
            ax5.set_xlabel("Epoch")
            ax5.set_ylabel("Counter")
            ax5.grid(True, alpha=0.3)
        else:
            ax5.axis("off")

        # Model Performance Summary
        total_epochs = len(epochs)
        actual_epochs = self.results.get("stopped_epoch", total_epochs)
        best_epoch = (
            self.results["val_accuracies"].index(max(self.results["val_accuracies"]))
            + 1
        )
        performance_data = [
            float(self.results["best_accuracy"]),
            float(self.results["val_losses"][-1]),
            int(actual_epochs),
            float(sum(self.results["epoch_times"]) / 60),
        ]
        performance_labels = [
            "Best Accuracy (%)",
            "Final Val Loss",
            "Total Epochs",
            "Total Time (min)",
        ]
        ax6.barh(
            performance_labels,
            performance_data,
            color=["green", "orange", "blue", "purple"],
            alpha=0.7,
        )
        ax6.set_title("Training Summary", fontsize=14, fontweight="bold")
        ax6.set_xlabel("Value")
        for i, v in enumerate(performance_data):
            ax6.text(v, i, f"{v:.2f}", va="center", fontweight="bold")
        plt.tight_layout()
        plt.show()

    def early_stopping_analysis(self):
        """Prints early stopping analysis."""
        print("\n🛑 Early Stopping Analysis:")
        print("=" * 50)
        if self.results.get("early_stopping_triggered", False):
            stopped_epoch = self.results.get("stopped_epoch", self.config["num_epochs"])
            planned_epochs = self.config["num_epochs"]
            epochs_saved = planned_epochs - stopped_epoch
            time_per_epoch = np.mean(self.results["epoch_times"])
            time_saved = epochs_saved * time_per_epoch / 60
            print(
                f"Early stopping triggered at epoch {stopped_epoch}/{planned_epochs}."
            )
            print(f"Epochs saved: {epochs_saved}")
            print(f"Time saved: ~{time_saved:.1f} minutes")
            print(f"Efficiency gain: {(epochs_saved/planned_epochs)*100:.1f}%")
        else:
            print("Training completed all planned epochs.")

    def training_statistics_summary(self):
        """Prints detailed training statistics."""
        print("\n📈 Detailed Training Statistics:")
        print("=" * 50)
        train_losses = self.results["train_losses"]
        val_accuracies = self.results["val_accuracies"]
        epoch_times = self.results["epoch_times"]
        print(
            f"- Loss Reduction: {train_losses[0]:.4f} → {train_losses[-1]:.4f} ({((train_losses[0] - train_losses[-1])/train_losses[0]*100):.1f}% improvement)"
        )
        print(
            f"- Best Accuracy: {max(val_accuracies):.2f}% (Epoch {val_accuracies.index(max(val_accuracies))+1})"
        )
        print(f"- Total Training Time: {sum(epoch_times)/60:.2f} minutes")
        print(f"- Average Time per Epoch: {np.mean(epoch_times):.2f}s")
        print(f"- Fastest Epoch: {min(epoch_times):.2f}s")
        print(f"- Slowest Epoch: {max(epoch_times):.2f}s")
        if "early_stopping_counter_history" in self.results:
            print(
                f"- Early Stopping Counter History: {self.results['early_stopping_counter_history']}"
            )

    def performance_metrics_table(self):
        """Prints a table of training metrics for the last 5 epochs and early stopping config."""
        epochs = range(1, len(self.results["train_losses"]) + 1)
        metrics_df = pd.DataFrame(
            {
                "Epoch": epochs,
                "Train_Loss": self.results["train_losses"],
                "Val_Loss": self.results["val_losses"],
                "Val_Accuracy": self.results["val_accuracies"],
                "Epoch_Time": self.results["epoch_times"],
                "Learning_Rate": self.results["learning_rates"],
                "ES_Counter": self.results.get(
                    "early_stopping_counter_history", [0] * len(epochs)
                ),
            }
        )
        metrics_df["Train_Loss"] = metrics_df["Train_Loss"].round(4)
        metrics_df["Val_Loss"] = metrics_df["Val_Loss"].round(4)
        metrics_df["Val_Accuracy"] = metrics_df["Val_Accuracy"].round(2)
        metrics_df["Epoch_Time"] = metrics_df["Epoch_Time"].round(2)
        print("\n📋 Training Metrics Table (Last 5 Epochs):")
        print(metrics_df.tail().to_string(index=False))
        print(f"\n🔍 Early Stopping Configuration:")
        print(f"- Patience: {self.config['early_stopping_patience']} epochs")
        print(
            f"- Delta: {self.config['early_stopping_delta']} (minimum improvement threshold)"
        )
        print(f"- Verbose: {self.config['early_stopping_verbose']}")
        print(f"- Monitoring: Validation Loss (lower is better)")

    def plot_confusion_matrix(self):
        """Plots confusion matrix and prints classification report and class imbalance analysis."""
        cm = confusion_matrix(
            self.results["final_targets"], self.results["final_predictions"]
        )
        cm_normalized = confusion_matrix(
            self.results["final_targets"],
            self.results["final_predictions"],
            normalize="true",
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax1,
            cbar_kws={"label": "Count"},
        )
        ax1.set_title("Confusion Matrix (Raw Counts)", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax2,
            cbar_kws={"label": "Proportion"},
        )
        ax2.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        plt.tight_layout()
        plt.show()

        print("\n📊 Detailed Classification Report:")
        print("=" * 80)
        report = classification_report(
            self.results["final_targets"],
            self.results["final_predictions"],
            target_names=self.class_names,
            digits=4,
        )
        print(report)

        print("\n⚖️ Class Imbalance Impact Analysis:")
        print("=" * 60)
        class_performance = {}
        minority_classes = []
        majority_classes = []
        median_count = np.median(self.class_counts)
        for i, class_name in enumerate(self.class_names):
            class_mask = np.array(self.results["final_targets"]) == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(
                    np.array(self.results["final_targets"])[class_mask],
                    np.array(self.results["final_predictions"])[class_mask],
                )
                class_precision, class_recall, class_f1, _ = (
                    precision_recall_fscore_support(
                        self.results["final_targets"],
                        self.results["final_predictions"],
                        labels=[i],
                        average=None,
                    )
                )
                class_performance[class_name] = {
                    "accuracy": class_acc * 100,
                    "precision": class_precision[0] * 100,
                    "recall": class_recall[0] * 100,
                    "f1": class_f1[0] * 100,
                    "count": int(self.class_counts[i]),
                }
                if self.class_counts[i] < median_count:
                    minority_classes.append(class_name)
                else:
                    majority_classes.append(class_name)
        minority_avg_acc = (
            np.mean([class_performance[cls]["accuracy"] for cls in minority_classes])
            if minority_classes
            else 0
        )
        majority_avg_acc = (
            np.mean([class_performance[cls]["accuracy"] for cls in majority_classes])
            if majority_classes
            else 0
        )
        print(f"\n📈 Class Group Performance:")
        print(f"Minority Classes ({len(minority_classes)}): {minority_classes}")
        print(f"Average Accuracy: {minority_avg_acc:.2f}%")
        print(f"Majority Classes ({len(majority_classes)}): {majority_classes}")
        print(f"Average Accuracy: {majority_avg_acc:.2f}%")
        print(
            f"Performance Gap: {abs(majority_avg_acc - minority_avg_acc):.2f}% {'(Majority better)' if majority_avg_acc > minority_avg_acc else '(Minority better)'}"
        )

        # Per-class accuracy and F1-score plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        classes = list(class_performance.keys())
        accuracies = [class_performance[cls]["accuracy"] for cls in classes]
        class_sizes = [class_performance[cls]["count"] for cls in classes]
        norm_sizes = [
            (
                (size - min(class_sizes)) / (max(class_sizes) - min(class_sizes))
                if max(class_sizes) > min(class_sizes)
                else 0.5
            )
            for size in class_sizes
        ]
        colors = plt.cm.RdYlBu_r(norm_sizes)
        bars1 = ax1.bar(classes, accuracies, color=colors, alpha=0.8, edgecolor="black")
        ax1.set_title(
            "Per-Class Accuracy (Color = Training Set Size)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Car Types")
        ax1.set_ylabel("Accuracy (%)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, acc, size in zip(bars1, accuracies, class_sizes):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%\n({size})",
                ha="center",
                fontweight="bold",
                fontsize=9,
            )
        f1_scores = [class_performance[cls]["f1"] for cls in classes]
        bars2 = ax2.bar(
            classes, f1_scores, color="lightcoral", alpha=0.7, edgecolor="darkred"
        )
        ax2.set_title("Per-Class F1-Score", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Car Types")
        ax2.set_ylabel("F1-Score (%)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{f1:.1f}%",
                ha="center",
                fontweight="bold",
            )
        plt.tight_layout()
        plt.show()

    def save_reports(self):
        """Saves training history, class performance, and early stopping summary to disk."""
        epochs = range(1, len(self.results["train_losses"]) + 1)
        history_df = pd.DataFrame(
            {
                "epoch": epochs,
                "train_loss": self.results["train_losses"],
                "val_loss": self.results["val_losses"],
                "val_accuracy": self.results["val_accuracies"],
                "epoch_time": self.results["epoch_times"],
                "early_stopping_counter": self.results.get(
                    "early_stopping_counter_history", [0] * len(epochs)
                ),
            }
        )
        reports_dir = "models/classification/reports"
        os.makedirs(reports_dir, exist_ok=True)
        config_info = (
            f"# Configuration: \n# use_weighted_loss={self.config['use_weighted_loss']}, use_class_balancing={self.config['use_class_balancing']}\n"
            f"# imbalance_ratio={self.class_counts.max()/self.class_counts.min():.2f}, early_stopping_patience={self.config['early_stopping_patience']}\n"
            f"# early_stopping_delta={self.config['early_stopping_delta']}, early_stopping_triggered={self.results.get('early_stopping_triggered', False)}\n"
            f"# stopped_epoch={self.results.get('stopped_epoch', 'N/A')}, best_val_loss={self.results.get('best_val_loss', 'N/A')}\n"
        )
        history_filename = os.path.join(
            reports_dir,
            f"training_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        with open(history_filename, "w") as f:
            f.write(config_info)
            history_df.to_csv(f, index=False)

        # Save class performance summary
        class_performance = {}
        median_count = np.median(self.class_counts)
        for i, class_name in enumerate(self.class_names):
            class_mask = np.array(self.results["final_targets"]) == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(
                    np.array(self.results["final_targets"])[class_mask],
                    np.array(self.results["final_predictions"])[class_mask],
                )
                class_precision, class_recall, class_f1, _ = (
                    precision_recall_fscore_support(
                        self.results["final_targets"],
                        self.results["final_predictions"],
                        labels=[i],
                        average=None,
                    )
                )
                class_performance[class_name] = {
                    "accuracy": class_acc * 100,
                    "precision": class_precision[0] * 100,
                    "recall": class_recall[0] * 100,
                    "f1": class_f1[0] * 100,
                    "count": int(self.class_counts[i]),
                    "samples_in_val": np.sum(class_mask),
                }
        performance_summary = pd.DataFrame(
            {
                "Class": list(class_performance.keys()),
                "Training_Size": [
                    class_performance[cls]["count"] for cls in class_performance
                ],
                "Accuracy": [
                    class_performance[cls]["accuracy"] for cls in class_performance
                ],
                "Precision": [
                    class_performance[cls]["precision"] for cls in class_performance
                ],
                "Recall": [
                    class_performance[cls]["recall"] for cls in class_performance
                ],
                "F1_Score": [class_performance[cls]["f1"] for cls in class_performance],
                "Val_Samples": [
                    class_performance[cls]["samples_in_val"]
                    for cls in class_performance
                ],
            }
        )
        performance_summary = performance_summary.round(2).sort_values("Training_Size")
        perf_filename = os.path.join(
            reports_dir,
            f"class_performance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        performance_summary.to_csv(perf_filename, index=False)

        # Save early stopping summary
        if self.results.get("early_stopping_triggered", False):
            es_summary = {
                "early_stopping_triggered": self.results.get(
                    "early_stopping_triggered", False
                ),
                "stopped_epoch": self.results.get("stopped_epoch", "N/A"),
                "planned_epochs": self.config["num_epochs"],
                "epochs_saved": self.config["num_epochs"]
                - self.results.get("stopped_epoch", self.config["num_epochs"]),
                "best_val_loss": self.results.get("best_val_loss"),
                "patience_used": self.config["early_stopping_patience"],
                "delta_threshold": self.config["early_stopping_delta"],
                "time_saved_minutes": (
                    self.config["num_epochs"]
                    - self.results.get("stopped_epoch", self.config["num_epochs"])
                )
                * np.mean(self.results["epoch_times"])
                / 60,
            }
            es_filename = os.path.join(
                reports_dir,
                f"early_stopping_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(es_filename, "w") as f:
                json.dump(es_summary, f, indent=2)
            print(f"\n🛑 Early stopping summary saved to: {es_filename}")
        print(f"\n💾 Training history saved to: {history_filename}")
        print(f"📊 Class performance saved to: {perf_filename}")
        print(f"🎯 Best model saved to: {self.config['result_path']}")
        if self.results.get("best_accuracy"):
            best_acc_path = self.config["result_path"].replace(".pth", "_best_acc.pth")
            print(f"🏆 Best accuracy model saved to: {best_acc_path}")
        print("\n🎉 Training Analysis Complete!")
        print("=" * 50)
        print(f"🎯 Final Summary:")
        stopped_epoch = self.results.get("stopped_epoch", self.config["num_epochs"])
        planned_epochs = self.config["num_epochs"]
        print(
            f"- Training completed in {stopped_epoch} epochs (planned: {planned_epochs})"
        )
        print(f"- Best validation accuracy: {max(self.results['val_accuracies']):.2f}%")
        print(
            f"- Early stopping: {'Activated' if self.results.get('early_stopping_triggered', False) else 'Not triggered'}"
        )
        print(
            f"- Total training time: {sum(self.results['epoch_times'])/60:.2f} minutes"
        )
        if self.results.get("early_stopping_triggered", False):
            time_saved = (
                (planned_epochs - stopped_epoch)
                * np.mean(self.results["epoch_times"])
                / 60
            )
            print(f"- Time saved by early stopping: {time_saved:.1f} minutes")
        max_count = self.class_counts.max().item()
        min_count = self.class_counts.min().item()
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 5:
            imbalance_level = "High"
        elif imbalance_ratio > 2:
            imbalance_level = "Moderate"
        else:
            imbalance_level = "Low"
        print(f"- Class imbalance: {imbalance_level} (ratio: {imbalance_ratio:.2f}x)")
        print(
            f"- Model saved with best validation loss: {self.results.get('best_val_loss', 'N/A'):.6f}"
        )

    def dataset_summary_report(self):
        """Prints and visualizes the dataset class distribution and summary."""
        train_counts = (
            self.trainer.train_dataset.class_counts
            if hasattr(self.trainer.train_dataset, "class_counts")
            else None
        )
        val_counts = (
            self.trainer.val_dataset.class_counts
            if hasattr(self.trainer.val_dataset, "class_counts")
            else None
        )

        if train_counts is None:
            train_counts = {cls: 0 for cls in self.class_names}
            for _, target in self.trainer.train_dataset:
                train_counts[self.class_names[target]] += 1
        if val_counts is None:
            val_counts = {cls: 0 for cls in self.class_names}
            for _, target in self.trainer.val_dataset:
                val_counts[self.class_names[target]] += 1
        print("=== Dataset Exploration ===")
        print("\nTraining set:")
        for class_name, count in sorted(train_counts.items()):
            print(f"  {class_name}: {count} images")
        print(f"  Total training images: {sum(train_counts.values())}")
        print("\nValidation set:")
        for class_name, count in sorted(val_counts.items()):
            print(f"  {class_name}: {count} images")
        print(f"  Total validation images: {sum(val_counts.values())}")
        # Visualize class distribution
        classes = list(train_counts.keys())
        train_values = list(train_counts.values())
        val_values = [val_counts.get(cls, 0) for cls in classes]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.bar(classes, train_values, color="skyblue", alpha=0.7)
        ax1.set_title(
            "Training Set - Class Distribution", fontsize=14, fontweight="bold"
        )
        ax1.set_xlabel("Car Types")
        ax1.set_ylabel("Number of Images")
        ax1.tick_params(axis="x", rotation=45)
        for i, v in enumerate(train_values):
            ax1.text(
                i, v + max(train_values) * 0.01, str(v), ha="center", fontweight="bold"
            )
        ax2.bar(classes, val_values, color="lightcoral", alpha=0.7)
        ax2.set_title(
            "Validation Set - Class Distribution", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Car Types")
        ax2.set_ylabel("Number of Images")
        ax2.tick_params(axis="x", rotation=45)
        for i, v in enumerate(val_values):
            ax2.text(
                i, v + max(val_values) * 0.01, str(v), ha="center", fontweight="bold"
            )
        plt.tight_layout()
        plt.show()
        # Create a summary dataframe
        df_summary = pd.DataFrame(
            {
                "Class": classes,
                "Training": train_values,
                "Validation": val_values,
                "Total": [train_values[i] + val_values[i] for i in range(len(classes))],
                "Train_Ratio": [
                    (
                        train_values[i] / (train_values[i] + val_values[i])
                        if (train_values[i] + val_values[i]) > 0
                        else 0
                    )
                    for i in range(len(classes))
                ],
            }
        )
        df_summary["Train_Ratio"] = df_summary["Train_Ratio"].round(3)
        print("\n=== Dataset Summary ===")
        print(df_summary.to_string(index=False))

    def plot_training_metrics(self):
        """Plot training and validation loss and accuracy curves, and print final values."""
        if not self.results:
            print("No training results to plot.")
            return
        metrics = self.results
        import matplotlib.pyplot as plt

        epochs = range(1, len(metrics["train_losses"]) + 1)
        plt.figure(figsize=(12, 5))
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, metrics["train_losses"], label="Train Loss")
        plt.plot(epochs, metrics["val_losses"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, metrics["train_accuracies"], label="Train Acc")
        plt.plot(epochs, metrics["val_accuracies"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(
            f"Final Train Loss: {metrics['train_losses'][-1]:.4f} | Final Val Loss: {metrics['val_losses'][-1]:.4f}"
        )
        print(
            f"Final Train Acc: {metrics['train_accuracies'][-1]:.2f}% | Final Val Acc: {metrics['val_accuracies'][-1]:.2f}%"
        )

    def full_report(self):
        """Runs all reporting and visualization methods in order."""
        self.plot_training_progress()
        self.early_stopping_analysis()
        self.training_statistics_summary()
        self.performance_metrics_table()
        self.plot_confusion_matrix()
        self.plot_training_metrics()

        self.save_reports()
