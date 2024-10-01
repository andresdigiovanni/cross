import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from cross import CrossTransformer
from cross.transformations.feature_engineering import MathematicalOperations
from cross.transformations.preprocessing import ScaleTransformation


def load_data():
    data = load_iris()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    x, y = df.drop(columns="target"), df["target"]

    return x, y


def evaluate_model(x, y, transformer=None):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kfold.split(x, y):
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]

        if transformer:
            x_train, x_test = transformer.fit_transform(x_train, x_test)
            x_test, y_test = transformer.transform(x_test, y_test)

        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)

    return np.mean(scores)


if __name__ == "__main__":
    x, y = load_data()

    # Evalute baseline model
    score = evaluate_model(x, y)
    print(f"Baseline score: {score}")

    # Define transformations
    transformer = CrossTransformer(
        [
            ScaleTransformation(
                transformation_options={
                    "sepal length (cm)": "min_max",
                    "sepal width (cm)": "min_max",
                    "petal length (cm)": "min_max",
                    "petal width (cm)": "min_max",
                }
            ),
            MathematicalOperations(
                operations_options=[
                    ("sepal length (cm)", "sepal width (cm)", "multiply"),
                    ("petal length (cm)", "petal width (cm)", "multiply"),
                ]
            ),
        ]
    )

    # Evalute model with transformations
    score = evaluate_model(x, y, transformer)
    print(f"Transformations score: {score}")
