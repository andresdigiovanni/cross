import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CastColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cast_options=None):
        self.cast_options = cast_options or {}

    def get_params(self, deep=True):
        return {"cast_options": self.cast_options}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        # No fitting needed for casting, just returns self
        return self

    def transform(self, X, y=None):
        X = X.copy()

        for column, dtype_to_cast in self.cast_options.items():
            if dtype_to_cast == "bool":
                X[column] = X[column].astype(bool)

            elif dtype_to_cast == "category":
                X[column] = X[column].astype(str)

            elif dtype_to_cast == "datetime":
                X[column] = pd.to_datetime(X[column])

            elif dtype_to_cast == "number":
                X[column] = pd.to_numeric(X[column], errors="coerce")

            elif dtype_to_cast == "timedelta":
                X[column] = pd.to_timedelta(X[column])

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
