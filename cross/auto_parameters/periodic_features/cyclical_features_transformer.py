from cross.transformations import CyclicalFeaturesTransformer
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class CyclicalFeaturesTransformerParamCalculator:
    VALID_PERIODS = {
        "_month": 12,
        "_day": 31,
        "_weekday": 7,
        "_hour": 24,
        "_minute": 60,
        "_second": 60,
    }

    PCT_UNIQUE_VALUES_THRESHOLD = 0.10

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        columns = dtypes.numerical_columns(x)
        transformation_options = {}

        logger.task_start("Detecting cyclical features")

        for column in columns:
            period = self._get_period(x, column)
            if period is not None:
                transformation_options[column] = period

        if transformation_options:
            logger.task_result(
                f"Cyclical features applied to {len(transformation_options)} column(s)"
            )
            return self._build_transformation_result(transformation_options)

        logger.warn("No cyclical features was applied to any column")
        return None

    def _get_period(self, df, column):
        column_lower = column.lower()

        for suffix, period in self.VALID_PERIODS.items():
            if column_lower.endswith(suffix):
                return period

        unique_values = df[column].dropna().unique()
        pct_unique_values = len(unique_values) / df.shape[0]

        if (
            len(unique_values) > 2
            and pct_unique_values < self.PCT_UNIQUE_VALUES_THRESHOLD
        ):
            return len(unique_values)

        return None

    def _build_transformation_result(self, transformation_options):
        transformer = CyclicalFeaturesTransformer(transformation_options)
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
