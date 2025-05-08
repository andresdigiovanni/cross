from collections import ChainMap

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import MissingValuesHandler
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class MissingValuesParamCalculator:
    def __init__(self):
        self.imputation_strategies = {
            "all": {
                "fill_0": {},
                "most_frequent": {},
            },
            "num": {
                "mean": {},
                "median": {},
                "knn": {"n_neighbors": [5]},
            },
            "cat": {},
        }

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        cat_columns = dtypes.categorical_columns(x)
        num_columns = dtypes.numerical_columns(x)
        x = x[cat_columns + num_columns]

        logger.task_start("Starting missing value imputation optimization")

        columns_with_nulls = self._get_columns_with_nulls(x)
        total_columns = len(columns_with_nulls)

        if total_columns == 0:
            logger.warn("No missing values found. Skipping imputation transformation.")
            return None

        best_transformation_options = {}
        best_n_neighbors = {}

        for i, column in enumerate(columns_with_nulls, start=1):
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column}'")

            best_strategy, best_params = self._find_best_strategy_for_column(
                x,
                y,
                model,
                scoring,
                direction,
                cv,
                groups,
                column,
                logger,
                is_num_column=(column in num_columns),
            )
            best_transformation_options[column] = best_strategy
            logger.task_result(f"Selected imputation for '{column}': {best_strategy}")

            if best_strategy == "knn":
                best_n_neighbors.update(best_params)

        logger.task_result(
            f"Imputation applied to {len(best_transformation_options)} column(s)"
        )

        return self._build_result(best_transformation_options, best_n_neighbors)

    def _get_columns_with_nulls(self, x):
        return x.columns[x.isnull().any()].tolist()

    def _find_best_strategy_for_column(
        self, x, y, model, scoring, direction, cv, groups, column, logger, is_num_column
    ):
        best_score = float("-inf") if direction == "maximize" else float("inf")
        best_strategy = None
        best_params = {}

        if is_num_column:
            imputation_strategies = ChainMap(
                self.imputation_strategies["all"], self.imputation_strategies["num"]
            )
        else:
            imputation_strategies = ChainMap(
                self.imputation_strategies["all"], self.imputation_strategies["cat"]
            )

        for strategy, params in imputation_strategies.items():
            if strategy == "knn":
                score, params = self._evaluate_knn_strategy(
                    x, y, model, scoring, direction, cv, groups, column, params, logger
                )
            else:
                score = self._evaluate_strategy(
                    x, y, model, scoring, cv, groups, column, strategy, logger
                )
                params = {}
                logger.progress(f"   ↪ Tried '{strategy}' → Score: {score:.4f}")

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_strategy = strategy
                best_params = params

        return best_strategy, best_params

    def _evaluate_knn_strategy(
        self, x, y, model, scoring, direction, cv, groups, column, params, logger
    ):
        best_score = float("-inf") if direction == "maximize" else float("inf")
        best_params = {}

        for n_neighbors in params["n_neighbors"]:
            score = self._evaluate_strategy(
                x, y, model, scoring, cv, groups, column, "knn", logger, n_neighbors
            )
            logger.progress(f"   ↪ Tried 'knn {n_neighbors}'  → Score: {score:.4f}")

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_params = {column: n_neighbors}

        return best_score, best_params

    def _evaluate_strategy(
        self,
        x,
        y,
        model,
        scoring,
        cv,
        groups,
        column,
        strategy,
        logger,
        n_neighbors=None,
    ):
        transformation_options = {column: strategy}
        knn_params = {column: n_neighbors} if n_neighbors else None

        missing_values_handler = MissingValuesHandler(
            transformation_options=transformation_options, n_neighbors=knn_params
        )

        return evaluate_model(x, y, model, scoring, cv, groups, missing_values_handler)

    def _build_result(self, transformation_options, n_neighbors):
        missing_values_handler = MissingValuesHandler(
            transformation_options=transformation_options, n_neighbors=n_neighbors
        )
        return {
            "name": missing_values_handler.__class__.__name__,
            "params": missing_values_handler.get_params(),
        }
