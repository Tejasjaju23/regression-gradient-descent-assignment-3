"""Dataset loading and reproducible train/test preparation."""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(test_size=0.20, random_state=42):
    """Load the real-world Diabetes regression dataset.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler
    """
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        dataset.feature_names,
        scaler,
    )
