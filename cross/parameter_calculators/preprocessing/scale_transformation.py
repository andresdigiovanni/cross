from tqdm import tqdm

from cross.parameter_calculators.shared import evaluate_model
from cross.transformations.preprocessing import ScaleTransformation
from cross.transformations.utils.dtypes import numerical_columns


class ScaleTransformationParamCalculator:
    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        columns = numerical_columns(x)
        transformation_options = {}
        scaler_options = ["min_max", "standard", "robust", "max_abs"]

        base_score = evaluate_model(x, y, model, scoring)

        for column in tqdm(columns, disable=(not verbose)):
            best_score = base_score
            best_params = {}

            for scaler in scaler_options:
                params = {column: scaler}
                scaler_tranformer = ScaleTransformation(params)
                score = evaluate_model(x, y, model, scoring, scaler_tranformer)

                has_improved = (direction == "maximize" and score > best_score) or (
                    direction != "maximize" and score < best_score
                )

                if has_improved:
                    best_score = score
                    best_params = params

            if best_params:
                transformation_options.update(best_params)

        if transformation_options:
            scale_transformation = ScaleTransformation(
                transformation_options=transformation_options
            )
            return {
                "name": scale_transformation.__class__.__name__,
                "params": scale_transformation.get_params(),
            }

        return None
