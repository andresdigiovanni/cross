import numpy as np


class MathematicalOperations:
    def __init__(self, operations_options=None):
        self.operations_options = operations_options or []

    def get_params(self):
        return {
            "operations_options": self.operations_options,
        }

    def fit(self, x, y=None):
        pass

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        for col1, col2, operation in self.operations_options:
            new_column = f"{col1}__{operation}__{col2}"

            if operation == "add":
                x_transformed[new_column] = x_transformed[col1] + x_transformed[col2]

            elif operation == "subtract":
                x_transformed[new_column] = x_transformed[col1] - x_transformed[col2]

            elif operation == "multiply":
                x_transformed[new_column] = x_transformed[col1] * x_transformed[col2]

            elif operation == "divide":
                x_transformed[new_column] = x_transformed[col1] / x_transformed[col2]

            elif operation == "modulus":
                x_transformed[new_column] = x_transformed[col1] % x_transformed[col2]

            elif operation == "power":
                x_transformed[new_column] = x_transformed[col1] ** x_transformed[col2]

            elif operation == "hypotenuse":
                x_transformed[new_column] = np.hypot(
                    x_transformed[col1], x_transformed[col2]
                )

            elif operation == "mean":
                x_transformed[new_column] = (
                    x_transformed[col1] + x_transformed[col2]
                ) / 2

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
