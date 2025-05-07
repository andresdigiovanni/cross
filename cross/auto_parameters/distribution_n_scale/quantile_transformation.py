from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import QuantileTransformation
from cross.transformations.utils import dtypes


class QuantileTransformationParamCalculator:
    QUANTILE_TRANSFORMATION_OPTIONS = ["uniform", "normal"]

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, verbose
    ):
        columns = dtypes.numerical_columns(x)
        transformation_options = {}
        base_score = evaluate_model(x, y, model, scoring, cv, groups)

        for column in tqdm(columns, disable=not verbose):
            best_params = self._find_best_quantile_transformation_for_column(
                x, y, model, scoring, base_score, column, direction, cv, groups
            )

            if best_params:
                transformation_options.update(best_params)

        if transformation_options:
            return self._build_transformation_result(transformation_options)

        return None

    def _find_best_quantile_transformation_for_column(
        self, x, y, model, scoring, base_score, column, direction, cv, groups
    ):
        best_score = base_score
        best_params = {}

        for transformation in self.QUANTILE_TRANSFORMATION_OPTIONS:
            params = {column: transformation}
            quantile_transformer = QuantileTransformation(params)
            score = evaluate_model(
                x, y, model, scoring, cv, groups, quantile_transformer
            )

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_params = params

        return best_params

    def _build_transformation_result(self, transformation_options):
        quantile_transformation = QuantileTransformation(
            transformation_options=transformation_options
        )
        return {
            "name": quantile_transformation.__class__.__name__,
            "params": quantile_transformation.get_params(),
        }
