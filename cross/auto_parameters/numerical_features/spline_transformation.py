from itertools import product

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import SplineTransformation
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class SplineTransformationParamCalculator:
    N_KNOTS_OPTIONS = [5, 10]
    DEGREE_OPTIONS = [3, 4]
    EXTRAPOLATION_OPTIONS = ["constant", "linear"]

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        columns = dtypes.numerical_columns(x)
        total_columns = len(columns)
        transformation_options = {}

        logger.task_start("Starting spline transformations search")
        base_score = evaluate_model(x, y, model, scoring, cv, groups)
        logger.baseline(f"Base score: {base_score:.4f}")

        for i, column in enumerate(columns, start=1):
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column}'")

            best_params = self._find_best_spline_transformation_for_column(
                x, y, model, scoring, base_score, column, direction, cv, groups, logger
            )

            if best_params:
                kwargs_str = (
                    f"extrapolation: {best_params[column]['extrapolation']}"
                    + f", degree: {best_params[column]['degree']}"
                    + f", n_knots: {best_params[column]['n_knots']}"
                )
                logger.task_result(
                    f"Selected spline transformation for '{column}': {kwargs_str}"
                )
                transformation_options.update(best_params)

        if transformation_options:
            logger.task_result(
                f"Spline transformations applied to {len(transformation_options)} column(s)"
            )
            return self._build_transformation_result(transformation_options)

        logger.warn("No spline transformations was applied to any column")
        return None

    def _find_best_spline_transformation_for_column(
        self, x, y, model, scoring, base_score, column, direction, cv, groups, logger
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
            logger.progress(
                f"   ↪ Tried {extrapolation=}, {degree=}, {n_knots=} → Score: {score:.4f}"
            )

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
