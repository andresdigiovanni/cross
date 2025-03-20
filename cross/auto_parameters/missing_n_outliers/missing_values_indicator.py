from cross.transformations import MissingValuesIndicator
from cross.transformations.utils.dtypes import categorical_columns, numerical_columns


class MissingValuesIndicatorParamCalculator:
    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, verbose
    ):
        cat_columns = categorical_columns(x)
        num_columns = numerical_columns(x)
        x = x[cat_columns + num_columns]

        columns_with_nulls = self._get_columns_with_nulls(x)
        if not columns_with_nulls:
            return None

        return self._build_result(columns_with_nulls)

    def _get_columns_with_nulls(self, x):
        return x.columns[x.isnull().any()].tolist()

    def _build_result(self, features):
        missing_values_indicator = MissingValuesIndicator(features=features)
        return {
            "name": missing_values_indicator.__class__.__name__,
            "params": missing_values_indicator.get_params(),
        }
