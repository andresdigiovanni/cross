from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelection(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def get_params(self, deep=True):
        return {"columns": self.columns}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        # No fitting required, maintaining compatibility with scikit-learn API
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Select only the specified columns from X
        X = X[self.columns]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
