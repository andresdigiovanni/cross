from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.transformations import OutliersHandler
from cross.transformations.utils.dtypes import numerical_columns


class OutliersParamCalculator:
    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        columns = numerical_columns(x)
        outlier_methods = self._get_outlier_methods()
        outlier_actions = ["cap", "median"]
        base_score = evaluate_model(x, y, model, scoring)

        best_params = {
            "handling_options": {},
            "thresholds": {},
            "lof_params": {},
            "iforest_params": {},
        }

        for column in tqdm(columns, disable=not verbose):
            best_column_params = self._find_best_params_for_column(
                x,
                y,
                model,
                scoring,
                direction,
                column,
                base_score,
                outlier_actions,
                outlier_methods,
            )

            if best_column_params:
                self._update_best_params(column, best_column_params, best_params)

        return self._build_outliers_handler(best_params)

    def _get_outlier_methods(self):
        return {
            "iqr": {"thresholds": [1.5]},
            "zscore": {"thresholds": [3.0]},
            "lof": {"n_neighbors": [20]},
            "iforest": {"contamination": [0.1]},
        }

    def _find_best_params_for_column(
        self, x, y, model, scoring, direction, column, base_score, actions, methods
    ):
        best_score = base_score
        best_params = {}
        combinations = self._generate_combinations(actions, methods)

        for action, method, param in combinations:
            if not self._has_outliers(x[column], method, param):
                continue

            kwargs = self._build_kwargs(column, action, method, param)
            score = evaluate_model(x, y, model, scoring, OutliersHandler(**kwargs))

            if self._is_score_improved(score, best_score, direction):
                best_score = score
                best_params = kwargs

        return best_params

    def _generate_combinations(self, actions, methods):
        combinations = [("none", "none", None)]

        for action in actions:
            for method, params in methods.items():
                if action == "cap" and method not in ["iqr", "zscore"]:
                    continue

                param_values = (
                    params.get("n_neighbors")
                    or params.get("contamination")
                    or params.get("thresholds")
                )
                combinations.extend(product([action], [method], param_values))

        return combinations

    def _has_outliers(self, column_data, method, param):
        if method in ["iqr", "zscore"]:
            return self._get_outliers_count(column_data, method, param) > 0

        if method in ["lof", "iforest"]:
            return self._get_outliers_count_ml(column_data, method, param) > 0

        return False

    def _get_outliers_count(self, column_data, method, param):
        if method == "iqr":
            q1, q3 = np.percentile(column_data, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - param * iqr, q3 + param * iqr

        else:  # zscore
            mean, std = column_data.mean(), column_data.std()
            lower, upper = mean - param * std, mean + param * std

        return column_data[(column_data < lower) | (column_data > upper)].shape[0]

    def _get_outliers_count_ml(self, column_data, method, param):
        model = (
            LocalOutlierFactor(n_neighbors=param)
            if method == "lof"
            else IsolationForest(contamination=param)
        )
        is_outlier = model.fit_predict(column_data.dropna().values.reshape(-1, 1)) == -1
        return is_outlier.sum()

    def _is_score_improved(self, score, best_score, direction):
        return (direction == "maximize" and score > best_score) or (
            direction == "minimize" and score < best_score
        )

    def _build_kwargs(self, column, action, method, param):
        kwargs = {"handling_options": {column: (action, method)}}

        if method == "lof":
            kwargs["lof_params"] = {column: {"n_neighbors": param}}

        elif method == "iforest":
            kwargs["iforest_params"] = {column: {"contamination": param}}

        else:
            kwargs["thresholds"] = {column: param}

        return kwargs

    def _update_best_params(self, column, best_column_params, best_params):
        action = best_column_params["handling_options"][column][0]

        if action != "none":
            best_params["handling_options"][column] = best_column_params[
                "handling_options"
            ][column]
            best_params["thresholds"].update(best_column_params.get("thresholds", {}))
            best_params["lof_params"].update(best_column_params.get("lof_params", {}))
            best_params["iforest_params"].update(
                best_column_params.get("iforest_params", {})
            )

    def _build_outliers_handler(self, best_params):
        if best_params["handling_options"]:
            outliers_handler = OutliersHandler(
                handling_options=best_params["handling_options"],
                thresholds=best_params["thresholds"],
                lof_params=best_params["lof_params"],
                iforest_params=best_params["iforest_params"],
            )
            return {
                "name": outliers_handler.__class__.__name__,
                "params": outliers_handler.get_params(),
            }

        return None
