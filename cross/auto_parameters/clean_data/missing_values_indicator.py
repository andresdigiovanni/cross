from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import MissingValuesIndicator
from cross.transformations.utils.dtypes import categorical_columns, numerical_columns


class MissingValuesIndicatorParamCalculator:
    def __init__(self):
        self.indicator_options = {
            "all": {"use_indicator": True},
        }

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, verbose
    ):
        cat_columns = categorical_columns(x)
        num_columns = numerical_columns(x)
        x = x[cat_columns + num_columns]

        columns_with_nulls = self._get_columns_with_nulls(x)
        if not columns_with_nulls:
            return None

        base_score = evaluate_model(x, y, model, scoring, cv, groups)
        features = []

        for column in tqdm(columns_with_nulls, disable=(not verbose)):
            include_feature = self._find_best_strategy_for_column(
                x,
                y,
                model,
                scoring,
                direction,
                cv,
                groups,
                column,
                base_score,
            )
            if include_feature:
                features.append(column)

        if not features:
            return None

        return self._build_result(features)

    def _get_columns_with_nulls(self, x):
        return x.columns[x.isnull().any()].tolist()

    def _find_best_strategy_for_column(
        self, x, y, model, scoring, direction, cv, groups, column, base_score
    ):
        indicator = MissingValuesIndicator(features=[column])
        score = evaluate_model(x, y, model, scoring, cv, groups, indicator)

        return is_score_improved(score, base_score, direction)

    def _build_result(self, features):
        missing_values_indicator = MissingValuesIndicator(features=features)
        return {
            "name": missing_values_indicator.__class__.__name__,
            "params": missing_values_indicator.get_params(),
        }
