import numpy as np


class CyclicalFeaturesTransformer:
    def __init__(self, columns_periods=None):
        self.columns_periods = columns_periods or {}

    def get_params(self):
        return {
            "columns_periods": self.columns_periods,
        }

    def fit(self, x, y=None):
        pass

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        for column, period in self.columns_periods.items():
            x_transformed[f"{column}_sin"] = np.sin(
                2 * np.pi * x_transformed[column] / period
            )
            x_transformed[f"{column}_cos"] = np.cos(
                2 * np.pi * x_transformed[column] / period
            )

        x_transformed = x_transformed.drop(columns=self.columns_periods.keys())

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
