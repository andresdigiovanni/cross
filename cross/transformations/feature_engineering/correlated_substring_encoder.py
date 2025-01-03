from sklearn.base import BaseEstimator, TransformerMixin


class CorrelatedSubstringEncoder(BaseEstimator, TransformerMixin):
    DEFAULT_VALUE = ""

    def __init__(self, substrings):
        self.substrings = substrings

    def get_params(self, deep=True):
        return {
            "substrings": self.substrings,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        return self  # No fitting necessary, but required for compatibility

    def transform(self, X, y=None):
        X = X.copy()

        for column, substrings in self.substrings.items():
            if column not in X.columns:
                print(f"Column {column} not in data.")
                continue

            new_column = f"{column}__corr_substring"
            X[new_column] = X[column].apply(
                lambda x: next(
                    (substring for substring in substrings if substring in x),
                    self.DEFAULT_VALUE,
                )
            )

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
