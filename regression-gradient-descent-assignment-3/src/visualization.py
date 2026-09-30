"""Visualization helpers."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_model_rmse(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(results["Model"], results["RMSE"])
    plt.xlabel("Model")
    plt.ylabel("Test RMSE")
    plt.title("Regression Model Comparison — Test RMSE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_model_r2(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(results["Model"], results["R2"])
    plt.xlabel("Model")
    plt.ylabel("Test R²")
    plt.title("Regression Model Comparison — Test R²")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_loss(history, learning_rate, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss (MSE / 2)")
    plt.title(f"Gradient Descent Convergence — learning rate={learning_rate}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, model_name, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    low = min(min(y_true), min(y_pred))
    high = max(max(y_true), max(y_pred))
    plt.plot([low, high], [low, high])
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.title(f"Actual vs Predicted — {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
