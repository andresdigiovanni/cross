import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutliersHandler:
    def __init__(
        self,
        handling_options=None,
        thresholds=None,
        lof_params=None,
        iforest_params=None,
        config=None,
    ):
        self.handling_options = handling_options or {}
        self.thresholds = thresholds or {}
        self.lof_params = lof_params or {}
        self.iforest_params = iforest_params or {}
        self.statistics = {}
        self.bounds = {}
        self.lof_results = {}
        self.iforest_results = {}

        if config:
            self.handling_options = config.get("handling_options", {})
            self.thresholds = config.get("thresholds", {})
            self.lof_params = config.get("lof_params", {})
            self.iforest_params = config.get("iforest_params", {})
            self.statistics = config.get("statistics", {})
            self.bounds = config.get("bounds", {})
            self.lof_results = config.get("lof_results", {})
            self.iforest_results = config.get("iforest_results", {})

    def get_params(self):
        params = {
            "handling_options": self.handling_options,
            "thresholds": self.thresholds,
            "lof_params": self.lof_params,
            "iforest_params": self.iforest_params,
            "statistics": self.statistics,
            "bounds": self.bounds,
            "lof_results": self.lof_results,
            "iforest_results": self.iforest_results,
        }
        return params

    def fit(self, df):
        self.statistics = {}
        self.bounds = {}
        self.lof_results = {}
        self.iforest_results = {}

        for column, (action, method) in self.handling_options.items():
            if method == "lof":
                self._apply_lof(df, column)

            elif method == "iforest":
                self._apply_iforest(df, column)

            lower_bound, upper_bound = self._calculate_bounds(df, column, method)
            self.bounds[column] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

            if action == "median":
                self.statistics[column] = df[column].median()

    def _apply_lof(self, df, column):
        lof = LocalOutlierFactor(**self.lof_params.get(column, {}))
        lof.fit(df[[column]])
        lof_scores = lof.negative_outlier_factor_
        self.lof_results[column] = lof_scores

    def _apply_iforest(self, df, column):
        iforest = IsolationForest(**self.iforest_params.get(column, {}))
        iforest.fit(df[[column]])
        iforest_scores = iforest.decision_function(df[[column]])
        self.iforest_results[column] = iforest_scores

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

        elif method == "lof":
            lof_scores = self.lof_results[column]
            non_outlier_values = df[lof_scores >= -self.thresholds.get(column, 1.5)][
                column
            ]

            lower_bound = non_outlier_values.min()
            upper_bound = non_outlier_values.max()

        elif method == "iforest":
            iforest_scores = self.iforest_results[column]
            non_outlier_values = df[iforest_scores >= self.thresholds.get(column, 0.0)][
                column
            ]

            lower_bound = non_outlier_values.min()
            upper_bound = non_outlier_values.max()

        else:
            lower_bound, upper_bound = 0, 0

        return lower_bound, upper_bound

    def transform(self, df):
        df_transformed = df.copy()

        for column, (action, method) in self.handling_options.items():
            lower_bound = self.bounds[column]["lower_bound"]
            upper_bound = self.bounds[column]["upper_bound"]

            if action == "remove":
                df_transformed = df_transformed[
                    (df_transformed[column] >= lower_bound)
                    & (df_transformed[column] <= upper_bound)
                ]

            elif action == "cap":
                df_transformed[column] = np.clip(
                    df_transformed[column], lower_bound, upper_bound
                )

            elif action == "median":
                df_transformed[column] = np.where(
                    (df_transformed[column] < lower_bound)
                    | (df_transformed[column] > upper_bound),
                    self.statistics[column],
                    df_transformed[column],
                )

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
