from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import OutliersHandler
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class OutliersParamCalculator:
    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        columns = dtypes.numerical_columns(x)
        total_columns = len(columns)
        outlier_methods = self._get_outlier_methods()
        outlier_actions = ["cap", "median"]

        best_params = {
            "transformation_options": {},
            "thresholds": {},
            "lof_params": {},
            "iforest_params": {},
        }

        logger.task_start("Starting outlier handling search")
        base_score = evaluate_model(x, y, model, scoring, cv, groups)
        logger.baseline(f"Base score: {base_score:.4f}")

        for i, column in enumerate(columns, start=1):
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column}'")

            best_column_params = self._find_best_params_for_column(
                x,
                y,
                model,
                scoring,
                direction,
                cv,
                groups,
                column,
                base_score,
                outlier_actions,
                outlier_methods,
                logger,
            )

            if best_column_params:
                kwargs_str = self._kwargs_to_string(best_column_params, column)
                logger.task_result(
                    f"Selected outlier handler for '{column}': {kwargs_str}"
                )
                self._update_best_params(column, best_column_params, best_params)

        transformation_options = best_params["transformation_options"]
        if transformation_options:
            logger.task_result(
                f"Outlier handler applied to {len(transformation_options)} column(s)"
            )
            return self._build_outliers_handler(best_params)

        logger.warn("No outlier handler was applied to any column")
        return self._build_outliers_handler(best_params)

    def _get_outlier_methods(self):
        return {
            "iqr": {"thresholds": [1.5, 3.0]},
            "zscore": {"thresholds": [2.5, 3.0]},
            "iforest": {"contamination": [0.05, 0.1]},
        }

    def _find_best_params_for_column(
        self,
        x,
        y,
        model,
        scoring,
        direction,
        cv,
        groups,
        column,
        base_score,
        actions,
        methods,
        logger,
    ):
        best_score = base_score
        best_params = {}
        combinations = self._generate_combinations(actions, methods)

        for action, method, param in combinations:
            if not self._has_outliers(x[column], method, param):
                continue

            kwargs = self._build_kwargs(column, action, method, param)
            score = evaluate_model(
                x, y, model, scoring, cv, groups, OutliersHandler(**kwargs)
            )
            kwargs_str = self._kwargs_to_string(kwargs, column)
            logger.progress(f"   ↪ Tried '{kwargs_str}' → Score: {score:.4f}")

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_params = kwargs

        return best_params

    def _generate_combinations(self, actions, methods):
        combinations = [("none", "none", None)]

        for method, params in methods.items():
            param_values = (
                params.get("n_neighbors")
                or params.get("contamination")
                or params.get("thresholds")
            )

            if method == "iforest":
                combinations.extend(product(["median"], [method], param_values))
            else:
                combinations.extend(product(actions, [method], param_values))

        return combinations

    def _has_outliers(self, column_data, method, param):
        if method in ["iqr", "zscore"]:
            return self._get_outliers_count(column_data, method, param) > 0

        if method in ["lof", "iforest"]:
            return self._get_outliers_count_ml(column_data, method, param) > 0

        return False

    def _get_outliers_count(self, column_data, method, param):
        if method == "iqr":
            q1, q3 = np.percentile(column_data.dropna(), [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - param * iqr, q3 + param * iqr
        else:  # zscore
            mean, std = column_data.mean(), column_data.std()
            lower, upper = mean - param * std, mean + param * std

        return column_data[(column_data < lower) | (column_data > upper)].shape[0]

    def _get_outliers_count_ml(self, column_data, method, param):
        clean_data = column_data.dropna().values.reshape(-1, 1)
        model = (
            LocalOutlierFactor(n_neighbors=param)
            if method == "lof"
            else IsolationForest(contamination=param, random_state=42)
        )
        is_outlier = model.fit_predict(clean_data) == -1
        return is_outlier.sum()

    def _build_kwargs(self, column, action, method, param):
        kwargs = {"transformation_options": {column: (action, method)}}

        if method == "lof":
            kwargs["lof_params"] = {column: {"n_neighbors": param}}
        elif method == "iforest":
            kwargs["iforest_params"] = {column: {"contamination": param}}
        else:
            kwargs["thresholds"] = {column: param}

        return kwargs

    def _kwargs_to_string(self, kwargs, column):
        action = kwargs["transformation_options"][column][0]
        method = kwargs["transformation_options"][column][1]

        if method == "lof":
            params = f"n_neighbors: {kwargs['lof_params'][column]['n_neighbors']}"

        elif method == "iforest":
            params = (
                f"contamination: {kwargs['iforest_params'][column]['contamination']}"
            )

        else:
            params = f"thresholds: {kwargs['thresholds'][column]}"

        return f"action: {action}, method: {method}, {params}"

    def _update_best_params(self, column, best_column_params, best_params):
        action = best_column_params["transformation_options"][column][0]

        if action != "none":
            best_params["transformation_options"][column] = best_column_params[
                "transformation_options"
            ][column]
            best_params["thresholds"].update(best_column_params.get("thresholds", {}))
            best_params["lof_params"].update(best_column_params.get("lof_params", {}))
            best_params["iforest_params"].update(
                best_column_params.get("iforest_params", {})
            )

    def _build_outliers_handler(self, best_params):
        if best_params["transformation_options"]:
            outliers_handler = OutliersHandler(
                transformation_options=best_params["transformation_options"],
                thresholds=best_params["thresholds"],
                lof_params=best_params["lof_params"],
                iforest_params=best_params["iforest_params"],
            )
            return {
                "name": outliers_handler.__class__.__name__,
                "params": outliers_handler.get_params(),
            }

        return None
