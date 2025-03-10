import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_periods=None):
        self.columns_periods = columns_periods or {}

    def get_params(self, deep=True):
        return {
            "columns_periods": self.columns_periods,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        return self  # No fitting necessary, but required for compatibility

    def transform(self, X, y=None):
        X = X.copy()

        for column, period in self.columns_periods.items():
            X[f"{column}_sin"] = np.sin(2 * np.pi * X[column] / period)
            X[f"{column}_cos"] = np.cos(2 * np.pi * X[column] / period)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
