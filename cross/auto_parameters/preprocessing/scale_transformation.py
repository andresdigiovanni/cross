from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import ScaleTransformation
from cross.transformations.utils.dtypes import numerical_columns


class ScaleTransformationParamCalculator:
    SCALER_OPTIONS = ["min_max", "standard", "robust", "max_abs"]

    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        columns = numerical_columns(x)
        transformation_options = {}
        base_score = evaluate_model(x, y, model, scoring)

        for column in tqdm(columns, disable=not verbose):
            best_params = self._find_best_scaler_for_column(
                x, y, model, scoring, base_score, column, direction
            )

            if best_params:
                transformation_options.update(best_params)

        if transformation_options:
            return self._build_transformation_result(transformation_options)

        return None

    def _find_best_scaler_for_column(
        self, x, y, model, scoring, base_score, column, direction
    ):
        best_score = base_score
        best_params = {}

        for scaler in self.SCALER_OPTIONS:
            params = {column: scaler}
            scale_transformer = ScaleTransformation(params)
            score = evaluate_model(x, y, model, scoring, scale_transformer)

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_params = params

        return best_params

    def _build_transformation_result(self, transformation_options):
        scale_transformation = ScaleTransformation(
            transformation_options=transformation_options
        )
        return {
            "name": scale_transformation.__class__.__name__,
            "params": scale_transformation.get_params(),
        }
