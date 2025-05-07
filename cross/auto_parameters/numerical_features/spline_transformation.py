from itertools import product

from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import SplineTransformation
from cross.transformations.utils import dtypes


class SplineTransformationParamCalculator:
    N_KNOTS_OPTIONS = [5, 10]
    DEGREE_OPTIONS = [3, 4, 5]
    EXTRAPOLATION_OPTIONS = ["constant", "linear", "continue", "periodic"]

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, verbose
    ):
        columns = dtypes.numerical_columns(x)
        transformation_options = {}
        base_score = evaluate_model(x, y, model, scoring, cv, groups)

        for column in tqdm(columns, disable=not verbose):
            best_params = self._find_best_spline_transformation_for_column(
                x, y, model, scoring, base_score, column, direction, cv, groups
            )

            if best_params:
                transformation_options.update(best_params)

        if transformation_options:
            return self._build_transformation_result(transformation_options)

        return None

    def _find_best_spline_transformation_for_column(
        self, x, y, model, scoring, base_score, column, direction, cv, groups
    ):
        best_score = base_score
        best_params = {}

        for n_knots, degree, extrapolation in product(
            self.N_KNOTS_OPTIONS, self.DEGREE_OPTIONS, self.EXTRAPOLATION_OPTIONS
        ):
            if extrapolation == "periodic" and degree >= n_knots:
                continue

            params = {
                column: {
                    "degree": degree,
                    "n_knots": n_knots,
                    "extrapolation": extrapolation,
                }
            }
            spline_transformer = SplineTransformation(params)
            score = evaluate_model(x, y, model, scoring, cv, groups, spline_transformer)

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_params = params

        return best_params

    def _build_transformation_result(self, transformation_options):
        spline_transformation = SplineTransformation(
            transformation_options=transformation_options
        )
        return {
            "name": spline_transformation.__class__.__name__,
            "params": spline_transformation.get_params(),
        }
