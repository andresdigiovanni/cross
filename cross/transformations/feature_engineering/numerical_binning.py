from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


class NumericalBinning(BaseEstimator, TransformerMixin):
    def __init__(self, binning_options=None):
        self.binning_options = binning_options or []

        self._binners = {}

    def get_params(self, deep=True):
        return {
            "binning_options": self.binning_options,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._binners = {}
        X_filled = X.copy()

        for column, strategy, n_bins in self.binning_options:
            if strategy == "none":
                continue

            X_filled[column] = X_filled[column].fillna(0)

            binner = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=strategy
            )
            binner.fit(X_filled[[column]])

            binner_name = f"{column}__{strategy}_{n_bins}"
            self._binners[binner_name] = binner

        return self

    def transform(self, X, y=None):
        X_transformed = X.copy().fillna(0)

        for column, strategy, n_bins in self.binning_options:
            if strategy == "none":
                continue

            binner_name = f"{column}__{strategy}_{n_bins}"
            binner = self._binners[binner_name]

            X_transformed[binner_name] = binner.transform(
                X_transformed[[column]]
            ).flatten()

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
