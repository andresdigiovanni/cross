from tqdm import tqdm

from cross.parameter_calculators.shared import evaluate_model
from cross.transformations.feature_engineering import CyclicalFeaturesTransformer
from cross.transformations.utils.dtypes import numerical_columns


class CyclicalFeaturesTransformerParamCalculator:
    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        columns = numerical_columns(x)
        columns_periods = {}

        baseline_score = evaluate_model(x, y, model, scoring)

        for column in tqdm(columns, disable=(not verbose)):
            period = self._get_period(x, column)

            if not period:
                continue

            cyclical_transformer = CyclicalFeaturesTransformer({column: period})
            score = evaluate_model(x, y, model, scoring, cyclical_transformer)

            if score > baseline_score:
                columns_periods[column] = period

        if columns_periods:
            datetime_transformer = CyclicalFeaturesTransformer(columns_periods)
            return {
                "name": datetime_transformer.__class__.__name__,
                "params": datetime_transformer.get_params(),
            }

        return None

    def _get_period(self, df, column):
        if column.lower().endswith("_month"):
            return 12

        elif column.lower().endswith("_day"):
            return 31

        elif column.lower().endswith("_weekday"):
            return 7

        elif column.lower().endswith("_hour"):
            return 24

        elif column.lower().endswith("_minute") or column.lower().endswith("_second"):
            return 60

        unique_values = df[column].dropna().unique()
        n_unique_values = len(unique_values)
        pct_unique_values = n_unique_values / df.shape[0]

        if n_unique_values > 1 and pct_unique_values < 0.10:
            return n_unique_values

        return None
