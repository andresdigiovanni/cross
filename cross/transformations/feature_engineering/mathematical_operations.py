import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MathematicalOperations(BaseEstimator, TransformerMixin):
    def __init__(self, operations_options=None):
        self.operations_options = operations_options or []

    def get_params(self, deep=True):
        return {
            "operations_options": self.operations_options,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        # No fitting required for this transformer, maintaining compatibility with scikit-learn API
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()

        for col1, col2, operation in self.operations_options:
            new_column = f"{col1}__{operation}__{col2}"

            if operation == "add":
                X_transformed[new_column] = X_transformed[col1] + X_transformed[col2]

            elif operation == "subtract":
                X_transformed[new_column] = X_transformed[col1] - X_transformed[col2]

            elif operation == "multiply":
                X_transformed[new_column] = X_transformed[col1] * X_transformed[col2]

            elif operation == "divide":
                X_transformed[new_column] = X_transformed[col1] / X_transformed[col2]
                X_transformed[new_column] = (
                    X_transformed[new_column]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )

            elif operation == "modulus":
                X_transformed[new_column] = X_transformed[col1] % X_transformed[col2]

            elif operation == "hypotenuse":
                X_transformed[new_column] = np.hypot(
                    X_transformed[col1], X_transformed[col2]
                )

            elif operation == "mean":
                X_transformed[new_column] = (
                    X_transformed[col1] + X_transformed[col2]
                ) / 2

            # Prevent fragmentation by explicitly copying the DataFrame
            X_transformed = X_transformed.copy()

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
