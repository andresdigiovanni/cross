from scipy.stats import skew
from tqdm import tqdm

from cross.transformations import NonLinearTransformation
from cross.transformations.utils.dtypes import numerical_columns


class NonLinearTransformationParamCalculator:
    SKEWNESS_THRESHOLD = 0.5

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, verbose
    ):
        best_transformation_options = {}
        columns = numerical_columns(x)

        for column in tqdm(columns, disable=not verbose):
            column_skewness = skew(x[column].dropna())

            if abs(column_skewness) < self.SKEWNESS_THRESHOLD:
                continue

            best_transformation_options[column] = "yeo_johnson"

        if best_transformation_options:
            return self._build_transformation_result(best_transformation_options)

        return None

    def _build_transformation_result(self, best_transformation_options):
        non_linear_transformation = NonLinearTransformation(
            transformation_options=best_transformation_options
        )
        return {
            "name": non_linear_transformation.__class__.__name__,
            "params": non_linear_transformation.get_params(),
        }
