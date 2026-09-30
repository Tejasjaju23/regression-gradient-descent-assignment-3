"""Complete Assignment 3 benchmark."""

from pathlib import Path
import sys
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset
from src.metrics import regression_metrics
from src.models import build_models
from src.gradient_descent import GradientDescentLinearRegression
from src.visualization import (
    plot_model_rmse,
    plot_model_r2,
    plot_loss,
    plot_actual_vs_predicted,
)

RESULT_DIR = ROOT / "experiments" / "results"
PLOT_DIR = ROOT / "outputs" / "plots"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, features, _ = load_dataset()

    print("Running baseline regression models...")
    rows = []
    predictions = {}

    for name, model in build_models().items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        elapsed = time.perf_counter() - start

        predictions[name] = pred
        metrics = regression_metrics(y_test, pred)
        rows.append({
            "Model": name,
            **metrics,
            "Training_Time_sec": elapsed,
        })

    # Gradient Descent learning-rate study
    gd_rows = []
    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1]

    for lr in learning_rates:
        start = time.perf_counter()
        gd = GradientDescentLinearRegression(
            learning_rate=lr,
            iterations=5000,
            tolerance=1e-8,
            min_iterations=100,
        )
        gd.fit(X_train, y_train)
        pred = gd.predict(X_test)
        elapsed = time.perf_counter() - start

        metrics = regression_metrics(y_test, pred)
        gd_rows.append({
            "Learning_Rate": lr,
            "Iterations": gd.n_iter_,
            "Converged": gd.converged_,
            "Initial_Loss": gd.loss_history_[0],
            "Final_Loss": gd.loss_history_[-1],
            "Test_MAE": metrics["MAE"],
            "Test_MSE": metrics["MSE"],
            "Test_RMSE": metrics["RMSE"],
            "Test_R2": metrics["R2"],
            "Training_Time_sec": elapsed,
        })

        safe_lr = str(lr).replace(".", "_")
        plot_loss(
            gd.loss_history_,
            lr,
            PLOT_DIR / f"gradient_descent_loss_lr_{safe_lr}.png",
        )

        if lr == 0.01:
            predictions["Gradient Descent Linear Regression"] = pred

    results = pd.DataFrame(rows)
    gd_results = pd.DataFrame(gd_rows)

    results.to_csv(RESULT_DIR / "model_results.csv", index=False)
    gd_results.to_csv(RESULT_DIR / "gradient_descent_results.csv", index=False)

    plot_model_rmse(results, PLOT_DIR / "model_rmse_comparison.png")
    plot_model_r2(results, PLOT_DIR / "model_r2_comparison.png")

    for name, pred in predictions.items():
        safe = (
            name.lower()
            .replace(" ", "_")
            .replace("+", "plus")
        )
        plot_actual_vs_predicted(
            y_test,
            pred,
            name,
            PLOT_DIR / f"actual_vs_predicted_{safe}.png",
        )

    print("\n=== MODEL RESULTS ===")
    print(results.to_string(index=False))

    print("\n=== GRADIENT DESCENT RESULTS ===")
    print(gd_results.to_string(index=False))

    print("\nResults saved to:")
    print(RESULT_DIR)
    print("\nPlots saved to:")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()
