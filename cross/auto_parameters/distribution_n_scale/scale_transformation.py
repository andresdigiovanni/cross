from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import ScaleTransformation
from cross.transformations.utils import dtypes


class ScaleTransformationParamCalculator:
    SCALER_OPTIONS = ["min_max", "standard", "robust"]
    QUANTILE_RANGE_OPTIONS = [1.0, 5.0, 10.0, 25.0]

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, verbose
    ):
        columns = dtypes.numerical_columns(x)
        base_score = evaluate_model(x, y, model, scoring, cv, groups)

        best_params = {
            "transformation_options": {},
            "quantile_range": {},
        }

        for column in tqdm(columns, disable=not verbose):
            best_column_params = self._find_best_scaler_for_column(
                x, y, model, scoring, base_score, column, direction, cv, groups
            )

            if best_column_params:
                best_params["transformation_options"].update(
                    best_column_params.get("transformation_options", {})
                )
                best_params["quantile_range"].update(
                    best_column_params.get("quantile_range", {})
                )

        if len(best_params["transformation_options"]):
            return self._build_transformation_result(best_params)

        return None

    def _find_best_scaler_for_column(
        self, x, y, model, scoring, base_score, column, direction, cv, groups
    ):
        best_score = base_score
        best_params = {}

        for scaler in self.SCALER_OPTIONS:
            if scaler == "robust":
                for quantile_range in self.QUANTILE_RANGE_OPTIONS:
                    quantile_range = (quantile_range, 100 - quantile_range)
                    params = {
                        "transformation_options": {column: scaler},
                        "quantile_range": {column: quantile_range},
                    }
                    scale_transformer = ScaleTransformation(**params)
                    score = evaluate_model(
                        x, y, model, scoring, cv, groups, scale_transformer
                    )

                    if is_score_improved(score, best_score, direction):
                        best_score = score
                        best_params = params

            else:
                params = {"transformation_options": {column: scaler}}
                scale_transformer = ScaleTransformation(**params)
                score = evaluate_model(
                    x, y, model, scoring, cv, groups, scale_transformer
                )

                if is_score_improved(score, best_score, direction):
                    best_score = score
                    best_params = params

        return best_params

    def _build_transformation_result(self, best_params):
        scale_transformation = ScaleTransformation(**best_params)
        return {
            "name": scale_transformation.__class__.__name__,
            "params": scale_transformation.get_params(),
        }
