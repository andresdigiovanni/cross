import numpy as np


class OutliersHandler:
    def __init__(self, handling_options=None, thresholds=None, config=None):
        self.handling_options = handling_options or {}
        self.thresholds = thresholds or {}
        self.statistics = {}
        self.bounds = {}

        if config:
            self.handling_options = config.get("handling_options", {})
            self.thresholds = config.get("thresholds", {})
            self.statistics = config.get("statistics", {})
            self.bounds = config.get("bounds", {})

    def get_params(self):
        params = {
            "handling_options": self.handling_options,
            "thresholds": self.thresholds,
            "statistics": self.statistics,
            "bounds": self.bounds,
        }
        return params

    def fit(self, df):
        self.statistics = {}
        self.bounds = {}

        for column, (action, method) in self.handling_options.items():
            lower_bound, upper_bound = self._calculate_bounds(df, column, method)
            self.bounds[column] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

            if action == "median":
                self.statistics[column] = df[column].median()

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
