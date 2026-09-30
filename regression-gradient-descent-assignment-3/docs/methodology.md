# Experimental Methodology

## Dataset

The scikit-learn Diabetes dataset is used as a real-world regression benchmark. It contains 442 samples, 10 baseline features and a continuous target.

## Split

An 80/20 train/test split with `random_state=42` is used.

## Preprocessing

StandardScaler is fitted only on the training set and then applied to the test set. This avoids using test-set information when estimating preprocessing parameters.

## Baseline Models

- Linear Regression
- Ridge Regression
- Random Forest Regression

## Optimization Study

Linear Regression is implemented from scratch using Batch Gradient Descent.

Learning rates tested:

- 0.001
- 0.003
- 0.01
- 0.03
- 0.1

Each experiment records the training-loss curve and final test metrics.

## Metrics

- MAE
- MSE
- RMSE
- R²

## Reproducibility

The random seed is fixed where randomness is used. The benchmark saves raw results as CSV files and generates plots from the same experiment.
