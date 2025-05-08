from cross.transformations import MissingValuesIndicator
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class MissingValuesIndicatorParamCalculator:
    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        cat_columns = dtypes.categorical_columns(x)
        num_columns = dtypes.numerical_columns(x)
        x = x[cat_columns + num_columns]

        logger.task_start("Starting missing value indicators")

        columns_with_nulls = self._get_columns_with_nulls(x)
        if not columns_with_nulls:
            logger.warn("No missing values found. Skipping indicator transformation.")
            return None

        logger.task_result(
            f"Selected {len(columns_with_nulls)} columns with missing values"
        )

        return self._build_result(columns_with_nulls)

    def _get_columns_with_nulls(self, x):
        return x.columns[x.isnull().any()].tolist()

    def _build_result(self, features):
        missing_values_indicator = MissingValuesIndicator(features=features)
        return {
            "name": missing_values_indicator.__class__.__name__,
            "params": missing_values_indicator.get_params(),
        }
