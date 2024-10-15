from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


class NumericalBinning(BaseEstimator, TransformerMixin):
    def __init__(self, binning_options=None, num_bins=None):
        self.binning_options = binning_options or {}
        self.num_bins = num_bins or {}

        self._binners = {}

    def get_params(self, deep=True):
        return {
            "binning_options": self.binning_options,
            "num_bins": self.num_bins,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        self._binners = {}
        X_filled = X.copy()

        for column, strategy in self.binning_options.items():
            X_filled[column] = X_filled[column].fillna(0)

            if strategy != "none":
                binner = KBinsDiscretizer(
                    n_bins=self.num_bins[column], encode="ordinal", strategy=strategy
                )
                binner.fit(X_filled[[column]])
                self._binners[column] = binner

        return self

    def transform(self, X, y=None):
        X_transformed = X.copy().fillna(0)

        for column, binner in self._binners.items():
            new_column = "{}__{}_{}".format(
                column, self.binning_options[column], self.num_bins[column]
            )
            X_transformed[new_column] = binner.transform(
                X_transformed[[column]]
            ).flatten()

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
