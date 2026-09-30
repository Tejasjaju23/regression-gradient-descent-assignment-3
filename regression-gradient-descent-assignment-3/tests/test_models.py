from src.data import load_dataset
from src.models import build_models


def test_baseline_models_fit_and_predict():
    X_train, X_test, y_train, y_test, _, _ = load_dataset()

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        assert len(prediction) == len(y_test)
