import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer

from cross.transformations.preprocessing import NonLinearTransformation
from cross.transformations.utils.dtypes import numerical_columns


class NonLinearTransformationParamCalculator:
    def calculate_best_params(self, x, y, problem_type, verbose):
        skewness_threshold = 0.5

        best_transformation_options = {}
        transformations = ["log", "exponential", "yeo_johnson"]

        columns = numerical_columns(x)

        for column in columns:
            column_skewness = skew(x[column].dropna())

            if abs(column_skewness) < skewness_threshold:
                continue

            best_score = abs(column_skewness)
            best_transformation = None

            for transformation in transformations:
                transformed_column = x[column].copy()

                if transformation == "log":
                    if (transformed_column <= 0).any():
                        continue

                    transformed_column = np.log1p(transformed_column)

                elif transformation == "exponential":
                    if (transformed_column < 0).any():
                        continue

                    transformed_column = np.exp(transformed_column)

                elif transformation == "yeo_johnson":
                    transformer = PowerTransformer(method="yeo-johnson")
                    transformed_column = transformer.fit_transform(
                        transformed_column.values.reshape(-1, 1)
                    ).flatten()

                transformed_skewness = skew(transformed_column)

                if abs(transformed_skewness) < best_score:
                    best_score = abs(transformed_skewness)
                    best_transformation = transformation

            if best_transformation:
                best_transformation_options[column] = best_transformation

        if best_transformation_options:
            non_linear_transformation = NonLinearTransformation(
                transformation_options=best_transformation_options
            )
            return {
                "name": non_linear_transformation.__class__.__name__,
                "params": non_linear_transformation.get_params(),
            }

        return None
