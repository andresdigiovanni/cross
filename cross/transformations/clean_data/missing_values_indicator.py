import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import MissingIndicator


class MissingValuesIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        self._indicator = None

    def get_params(self, deep=True):
        return {"features": self.features}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._indicator = MissingIndicator(features="all", sparse=False)
        self._indicator.fit(X[self.features])
        return self

    def transform(self, X, y=None):
        X = X.copy()

        transformed_array = self._indicator.transform(X[self.features])
        columns = [f"{col}__is_missing" for col in self.features]

        encoded_df = pd.DataFrame(
            transformed_array.astype(int),
            columns=columns,
            index=X.index,
        )
        X = pd.concat([X, encoded_df], axis=1)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
