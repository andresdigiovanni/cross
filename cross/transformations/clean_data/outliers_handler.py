import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutliersHandler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        handling_options=None,
        thresholds=None,
        lof_params=None,
        iforest_params=None,
    ):
        self.handling_options = handling_options
        self.thresholds = thresholds
        self.lof_params = lof_params
        self.iforest_params = iforest_params

        self._statistics = {}
        self._bounds = {}
        self._lof_results = {}
        self._iforest_results = {}

    def get_params(self, deep=True):
        return {
            "handling_options": self.handling_options,
            "thresholds": self.thresholds,
            "lof_params": self.lof_params,
            "iforest_params": self.iforest_params,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._statistics = {}
        self._bounds = {}
        self._lof_results = {}
        self._iforest_results = {}

        for column, (action, method) in self.handling_options.items():
            # Specific methods fit
            if method == "lof":
                self._apply_lof(X, column)

            elif method == "iforest":
                self._apply_iforest(X, column)

            elif method in ["iqr", "zscore"]:
                lower_bound, upper_bound = self._calculate_bounds(X, column, method)
                self._bounds[column] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }

            # Specific actions fit
            if action == "median":
                self._statistics[column] = X[column].median()

        return self

    def _apply_lof(self, df, column):
        lof = LocalOutlierFactor(**self.lof_params[column])
        lof.fit(df[[column]])
        lof_scores = lof.negative_outlier_factor_
        self._lof_results[column] = lof_scores

    def _apply_iforest(self, df, column):
        iforest = IsolationForest(**self.iforest_params[column])
        iforest.fit(df[[column]])
        iforest_scores = iforest.decision_function(df[[column]])
        self._iforest_results[column] = iforest_scores

    def _calculate_bounds(self, df, column, method):
        if method == "iqr":
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - self.thresholds[column] * iqr
            upper_bound = q3 + self.thresholds[column] * iqr

        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()

            lower_bound = mean - self.thresholds[column] * std
            upper_bound = mean + self.thresholds[column] * std

        else:
            lower_bound, upper_bound = 0, 0

        return lower_bound, upper_bound

    def transform(self, X, y=None):
        X = X.copy()

        for column, (action, method) in self.handling_options.items():
            if method in ["iqr", "zscore"]:
                lower_bound = self._bounds[column]["lower_bound"]
                upper_bound = self._bounds[column]["upper_bound"]

            if action == "cap":
                X[column] = np.clip(X[column], lower_bound, upper_bound)

            elif action == "median":
                if method == "lof":
                    lof = LocalOutlierFactor(
                        n_neighbors=self.lof_params[column]["n_neighbors"]
                    )
                    y_pred = lof.fit_predict(X[[column]])
                    outliers = y_pred == -1

                elif method == "iforest":
                    iforest = IsolationForest(
                        contamination=self.iforest_params[column]["contamination"]
                    )
                    y_pred = iforest.fit_predict(X[[column]].dropna())
                    outliers = y_pred == -1

                elif method in ["iqr", "zscore"]:
                    outliers = (X[column] < lower_bound) | (X[column] > upper_bound)

                X[column] = np.where(
                    outliers,
                    self._statistics[column],
                    X[column],
                )

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
