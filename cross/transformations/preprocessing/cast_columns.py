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
        X_transformed = X.copy()

        for column, dtype_to_cast in self.cast_options.items():
            if dtype_to_cast == "bool":
                X_transformed[column] = X_transformed[column].astype(bool)

            elif dtype_to_cast == "category":
                X_transformed[column] = X_transformed[column].astype(str)

            elif dtype_to_cast == "datetime":
                X_transformed[column] = pd.to_datetime(X_transformed[column])

            elif dtype_to_cast == "number":
                X_transformed[column] = pd.to_numeric(
                    X_transformed[column], errors="coerce"
                )

            elif dtype_to_cast == "timedelta":
                X_transformed[column] = pd.to_timedelta(X_transformed[column])

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
