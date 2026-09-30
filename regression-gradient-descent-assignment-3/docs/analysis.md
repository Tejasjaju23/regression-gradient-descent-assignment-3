# Critical Analysis Guide

## Model Selection

Linear Regression is the interpretable baseline. Ridge adds regularization, while Random Forest provides a non-linear alternative.

## Metrics

Use MAE, RMSE and R² together rather than relying on one metric.

- MAE: average absolute error.
- RMSE: penalizes large errors more strongly.
- R²: proportion of variance explained relative to a mean baseline.

## Gradient Descent

Analyze:

1. Whether loss decreases monotonically or approximately monotonically.
2. How many iterations are required for convergence.
3. Whether the learning rate is too small, appropriate or too large.
4. Whether the final Gradient Descent model approaches the ordinary Linear Regression solution.
5. How optimization behavior differs from test-set generalization.

## Required evidence

Use:

- `model_results.csv`
- `gradient_descent_results.csv`
- loss plots
- model comparison plots
- actual-vs-predicted plots

Do not invent numbers. All numerical claims in the final report should be traceable to generated experimental output.
