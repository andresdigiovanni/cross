import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm

from cross.parameter_calculators.shared import evaluate_model
from cross.transformations.preprocessing import NonLinearTransformation
from cross.transformations.utils.dtypes import numerical_columns


class NonLinearTransformationParamCalculator:
    SKEWNESS_THRESHOLD = 0.5
    TRANSFORMATIONS = ["log", "exponential", "yeo_johnson"]

    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        best_transformation_options = {}
        base_score = evaluate_model(x, y, model, scoring)
        columns = numerical_columns(x)

        for column in tqdm(columns, disable=not verbose):
            column_skewness = skew(x[column].dropna())

            if abs(column_skewness) < self.SKEWNESS_THRESHOLD:
                continue

            best_transformation = self._find_best_transformation(
                x[column], column_skewness
            )

            if best_transformation:
                score = evaluate_model(
                    x,
                    y,
                    model,
                    scoring,
                    NonLinearTransformation({column: best_transformation}),
                )

                if self._is_score_improved(score, base_score, direction):
                    best_transformation_options[column] = best_transformation

        if best_transformation_options:
            return self._build_transformation_result(best_transformation_options)

        return None

    def _find_best_transformation(self, column_data, column_skewness):
        best_score = abs(column_skewness)
        best_transformation = None

        for transformation in self.TRANSFORMATIONS:
            transformed_column = self._apply_transformation(
                column_data.copy(), transformation
            )

            if transformed_column is None:
                continue

            transformed_skewness = skew(transformed_column)

            if abs(transformed_skewness) < best_score:
                best_score = abs(transformed_skewness)
                best_transformation = transformation

        return best_transformation

    def _apply_transformation(self, column_data, transformation):
        if transformation == "log":
            if (column_data <= 0).any():
                return None

            return np.log1p(column_data)

        elif transformation == "exponential":
            if (column_data < 0).any():
                return None

            return np.exp(column_data)

        elif transformation == "yeo_johnson":
            transformer = PowerTransformer(method="yeo-johnson")

            return transformer.fit_transform(
                column_data.values.reshape(-1, 1)
            ).flatten()

    def _is_score_improved(self, score, best_score, direction):
        return (direction == "maximize" and score > best_score) or (
            direction == "minimize" and score < best_score
        )

    def _build_transformation_result(self, best_transformation_options):
        non_linear_transformation = NonLinearTransformation(
            transformation_options=best_transformation_options
        )
        return {
            "name": non_linear_transformation.__class__.__name__,
            "params": non_linear_transformation.get_params(),
        }
