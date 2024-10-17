from cross.transformations.feature_engineering import DateTimeTransformer
from cross.transformations.utils.dtypes import datetime_columns


class DateTimeTransformerParamCalculator:
    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        columns = datetime_columns(x)

        if not columns:
            return None

        datetime_transformer = DateTimeTransformer(columns)
        return {
            "name": datetime_transformer.__class__.__name__,
            "params": datetime_transformer.get_params(),
        }
