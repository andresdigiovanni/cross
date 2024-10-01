import numpy as np

from cross.transformations.preprocessing import ScaleTransformation
from cross.transformations.utils.dtypes import numerical_columns


class ScaleTransformationParamCalculator:
    def calculate_best_params(self, x, y=None, problem_type=None):
        columns = numerical_columns(x)
        transformation_options = {}

        for column in columns:
            col_data = x[column].dropna()

            if self._has_outliers(col_data):
                transformation = "robust"

            else:
                transformation = "min_max"

            transformation_options[column] = transformation

        if transformation_options:
            scale_transformation = ScaleTransformation(
                transformation_options=transformation_options
            )
            return {
                "name": scale_transformation.__class__.__name__,
                "params": scale_transformation.get_params(),
            }

        return None

    def _has_outliers(self, data, threshold=1.5):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        return ((data < lower_bound) | (data > upper_bound)).any()
