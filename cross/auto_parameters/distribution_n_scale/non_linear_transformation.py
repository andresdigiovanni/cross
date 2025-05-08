from scipy.stats import skew

from cross.transformations import NonLinearTransformation
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class NonLinearTransformationParamCalculator:
    SKEWNESS_THRESHOLD = 0.5

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        best_transformation_options = {}
        columns = dtypes.numerical_columns(x)
        total_columns = len(columns)

        logger.task_start("Starting non-linear transformation search")

        for i, column in enumerate(columns, start=1):
            logger.task_update(f"[{i}/{total_columns}] Checking column: '{column}'")

            column_skewness = skew(x[column].dropna())
            logger.progress(f"   ↪ Skewness: {column_skewness:.4f}")

            if abs(column_skewness) < self.SKEWNESS_THRESHOLD:
                continue

            best_transformation_options[column] = "yeo_johnson"
            logger.task_result(f"Selected 'yeo_johnson' for '{column}'")

        if best_transformation_options:
            logger.task_result(
                f"Non-linear transformation applied to {len(best_transformation_options)} column(s)"
            )
            return self._build_transformation_result(best_transformation_options)

        logger.warn("No columns required non-linear transformation")
        return None

    def _build_transformation_result(self, best_transformation_options):
        non_linear_transformation = NonLinearTransformation(
            transformation_options=best_transformation_options
        )
        return {
            "name": non_linear_transformation.__class__.__name__,
            "params": non_linear_transformation.get_params(),
        }
