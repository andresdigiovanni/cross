import numpy as np


class MathematicalOperations:
    def __init__(self, operations_options=None, config=None):
        self.operations_options = operations_options or {}

        if config:
            self.operations_options = config.get("operations_options", {})

    def get_params(self):
        params = {
            "operations_options": self.operations_options,
        }
        return params

    def fit(self, df):
        pass

    def transform(self, df):
        df_transformed = df.copy()

        for col1, col2, operation in self.operations_options:
            new_column = f"{col1}__{operation}__{col2}"

            if operation == "add":
                df_transformed[new_column] = df_transformed[col1] + df_transformed[col2]

            elif operation == "subtract":
                df_transformed[new_column] = df_transformed[col1] - df_transformed[col2]

            elif operation == "multiply":
                df_transformed[new_column] = df_transformed[col1] * df_transformed[col2]

            elif operation == "divide":
                df_transformed[new_column] = df_transformed[col1] / df_transformed[col2]

            elif operation == "modulus":
                df_transformed[new_column] = df_transformed[col1] % df_transformed[col2]

            elif operation == "power":
                df_transformed[new_column] = (
                    df_transformed[col1] ** df_transformed[col2]
                )

            elif operation == "hypotenuse":
                df_transformed[new_column] = np.hypot(
                    df_transformed[col1], df_transformed[col2]
                )

            elif operation == "mean":
                df_transformed[new_column] = (
                    df_transformed[col1] + df_transformed[col2]
                ) / 2

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
