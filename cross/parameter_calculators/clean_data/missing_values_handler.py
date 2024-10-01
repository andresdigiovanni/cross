from cross.parameter_calculators.shared import evaluate_model
from cross.transformations.clean_data import MissingValuesHandler
from cross.transformations.utils.dtypes import numerical_columns


class MissingValuesParamCalculator:
    def __init__(self):
        self.imputation_strategies = {
            "fill_0": {},
            "fill_mean": {},
            "fill_median": {},
            "fill_mode": {},
            "fill_knn": {"n_neighbors": [3, 5, 7]},
            "most_frequent": {},
        }

    def calculate_best_params(self, x, y, problem_type):
        num_columns = numerical_columns(x)
        x = x[num_columns]

        columns_with_nulls = self._get_columns_with_nulls(x)

        if not columns_with_nulls:
            return None

        best_handling_options = {}
        best_n_neighbors = {}

        for column in columns_with_nulls:
            best_strategy, best_params = self._find_best_strategy_for_column(
                x, y, problem_type, column
            )
            best_handling_options[column] = best_strategy

            if best_strategy == "fill_knn":
                best_n_neighbors.update(best_params)

        return self._build_result(best_handling_options, best_n_neighbors)

    def _get_columns_with_nulls(self, x):
        return x.columns[x.isnull().any()].tolist()

    def _find_best_strategy_for_column(self, x, y, problem_type, column):
        best_score = -float("inf")
        best_strategy = None
        best_params = {}

        for strategy, params in self.imputation_strategies.items():
            if strategy == "fill_knn":
                for n_neighbors in params["n_neighbors"]:
                    score = self._evaluate_strategy(
                        x, y, problem_type, column, strategy, n_neighbors
                    )
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
                        best_params = {column: n_neighbors}
            else:
                score = self._evaluate_strategy(x, y, problem_type, column, strategy)
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
                    best_params = {}

        return best_strategy, best_params

    def _evaluate_strategy(
        self, x, y, problem_type, column, strategy, n_neighbors=None
    ):
        handling_options = {column: strategy}
        knn_params = {column: n_neighbors} if n_neighbors else None

        missing_values_handler = MissingValuesHandler(
            handling_options=handling_options, n_neighbors=knn_params
        )

        return evaluate_model(x, y, problem_type, missing_values_handler)

    def _build_result(self, best_handling_options, best_n_neighbors):
        missing_values_handler = MissingValuesHandler(
            handling_options=best_handling_options, n_neighbors=best_n_neighbors
        )
        return {
            "name": missing_values_handler.__class__.__name__,
            "params": missing_values_handler.get_params(),
        }
