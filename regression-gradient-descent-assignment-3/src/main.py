"""Quick demonstration of Assignment 3."""

from pathlib import Path
import pandas as pd

from src.data import load_dataset
from src.metrics import regression_metrics
from src.gradient_descent import GradientDescentLinearRegression
from src.models import build_models


def main():
    X_train, X_test, y_train, y_test, feature_names, _ = load_dataset()

    print("=" * 65)
    print(" ASSIGNMENT 3 — REGRESSION & GRADIENT DESCENT")
    print("=" * 65)
    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Features         : {len(feature_names)}")

    rows = []

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics = regression_metrics(y_test, prediction)
        rows.append({"Model": name, **metrics})

    gd = GradientDescentLinearRegression(
        learning_rate=0.01, iterations=5000
    )
    gd.fit(X_train, y_train)
    prediction = gd.predict(X_test)
    rows.append({
        "Model": "Gradient Descent Linear Regression",
        **regression_metrics(y_test, prediction)
    })

    results = pd.DataFrame(rows)
    print("\nTest-set performance:")
    print(results.to_string(index=False))

    print(f"\nGradient Descent iterations: {gd.n_iter_}")
    print(f"Gradient Descent converged : {gd.converged_}")
    print(f"Final training loss        : {gd.loss_history_[-1]:.6f}")


if __name__ == "__main__":
    main()
