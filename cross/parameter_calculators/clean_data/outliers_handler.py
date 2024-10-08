from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from cross.parameter_calculators.shared import evaluate_model
from cross.transformations.clean_data import OutliersHandler
from cross.transformations.utils.dtypes import numerical_columns


class OutliersParamCalculator:
    def calculate_best_params(self, x, y, problem_type, verbose):
        if y is None:
            return None

        columns = numerical_columns(x)
        outlier_methods = self._get_outlier_methods()
        outlier_actions = ["cap", "median"]

        best_handling_options = {}
        best_thresholds = {}
        best_lof_params = {}
        best_iforest_params = {}

        for column in tqdm(columns, disable=(not verbose)):
            best_params = self._find_best_params_for_column(
                x, y, problem_type, column, outlier_actions, outlier_methods
            )

            if best_params:
                self._update_best_params(
                    best_params,
                    column,
                    best_handling_options,
                    best_thresholds,
                    best_lof_params,
                    best_iforest_params,
                )

        return self._build_outliers_handler(
            best_handling_options, best_thresholds, best_lof_params, best_iforest_params
        )

    def _get_outlier_methods(self):
        return {
            "iqr": {"thresholds": [1.5]},
            "zscore": {"thresholds": [3.0]},
            "lof": {"n_neighbors": [20]},
            "iforest": {"contamination": [0.1]},
        }

    def _find_best_params_for_column(
        self, x, y, problem_type, column, outlier_actions, outlier_methods
    ):
        best_score = -float("inf")
        best_params = {}

        combinations = self._generate_combinations(outlier_actions, outlier_methods)

        for action, method, param in combinations:
            rows_affected = float("inf")

            if method == "iqr":
                q1, q3 = np.percentile(x[column], [25, 75])
                iqr = q3 - q1

                lower_bound = q1 - param * iqr
                upper_bound = q3 + param * iqr

                rows_affected = x[
                    (x[column] < lower_bound) | (x[column] > upper_bound)
                ].shape[0]

            elif method == "zscore":
                mean = x[column].mean()
                std = x[column].std()

                lower_bound = mean - param * std
                upper_bound = mean + param * std

                rows_affected = x[
                    (x[column] < lower_bound) | (x[column] > upper_bound)
                ].shape[0]

            elif method == "lof":
                lof = LocalOutlierFactor(n_neighbors=param)
                y_pred = lof.fit_predict(x[[column]].dropna())
                is_outlier = y_pred == -1

                rows_affected = is_outlier.sum()

            elif method == "iforest":
                iforest = IsolationForest(contamination=param)
                y_pred = iforest.fit_predict(x[[column]].dropna())
                is_outlier = y_pred == -1

                rows_affected = is_outlier.sum()

            if not rows_affected:
                continue

            kwargs = self._build_kwargs(column, action, method, param)
            score = evaluate_model(x, y, problem_type, OutliersHandler(**kwargs))

            if score > best_score:
                best_score = score
                best_params = kwargs

        return best_params

    def _generate_combinations(self, outlier_actions, outlier_methods):
        combinations = [("none", "none", None)]

        for action in outlier_actions:
            for method, params in outlier_methods.items():
                if method == "lof":
                    combinations.extend(
                        product([action], [method], params["n_neighbors"])
                    )

                elif method == "iforest":
                    combinations.extend(
                        product([action], [method], params["contamination"])
                    )

                else:
                    combinations.extend(
                        product([action], [method], params["thresholds"])
                    )

        return combinations

    def _build_kwargs(self, column, action, method, param):
        kwargs = {"handling_options": {column: (action, method)}}

        if method == "lof":
            kwargs["lof_params"] = {column: {"n_neighbors": param}}

        elif method == "iforest":
            kwargs["iforest_params"] = {column: {"contamination": param}}

        else:
            kwargs["thresholds"] = {column: param}

        return kwargs

    def _update_best_params(
        self,
        best_params,
        column,
        best_handling_options,
        best_thresholds,
        best_lof_params,
        best_iforest_params,
    ):
        best_action = best_params["handling_options"][column][0]

        if best_action != "none":
            best_handling_options[column] = best_params["handling_options"][column]

            if "lof_params" in best_params:
                best_lof_params.update(best_params["lof_params"])

            elif "iforest_params" in best_params:
                best_iforest_params.update(best_params["iforest_params"])

            else:
                best_thresholds.update(best_params["thresholds"])

    def _build_outliers_handler(
        self,
        best_handling_options,
        best_thresholds,
        best_lof_params,
        best_iforest_params,
    ):
        if best_handling_options:
            outliers_handler = OutliersHandler(
                handling_options=best_handling_options,
                thresholds=best_thresholds,
                lof_params=best_lof_params,
                iforest_params=best_iforest_params,
            )
            return {
                "name": outliers_handler.__class__.__name__,
                "params": outliers_handler.get_params(),
            }

        return None
