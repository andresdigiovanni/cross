from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import QuantileTransformation
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class QuantileTransformationParamCalculator:
    QUANTILE_TRANSFORMATION_OPTIONS = ["uniform", "normal"]

    def calculate_best_params(
        self,
        x,
        y,
        model,
        scoring,
        direction,
        cv,
        groups,
        logger: VerboseLogger,
    ):
        columns = dtypes.numerical_columns(x)
        transformation_options = {}
        total_columns = len(columns)

        logger.task_start("Starting quantile transformation parameter search")
        base_score = evaluate_model(x, y, model, scoring, cv, groups)
        logger.baseline(f"Base score: {base_score:.4f}")

        for i, column in enumerate(columns, start=1):
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column}'")

            best_params = self._find_best_quantile_transformation_for_column(
                x, y, model, scoring, base_score, column, direction, cv, groups, logger
            )

            if best_params:
                transformation_options.update(best_params)
                logger.task_result(
                    f"Selected transformation for '{column}': {list(best_params.values())[0]}"
                )

        if transformation_options:
            logger.task_result(
                f"Quantile transformation applied to {len(transformation_options)} column(s)"
            )
            return self._build_transformation_result(transformation_options)

        logger.warn("No quantile transformation was applied to any column")
        return None

    def _find_best_quantile_transformation_for_column(
        self,
        x,
        y,
        model,
        scoring,
        base_score,
        column,
        direction,
        cv,
        groups,
        logger: VerboseLogger,
    ):
        best_score = base_score
        best_params = {}

        for transformation in self.QUANTILE_TRANSFORMATION_OPTIONS:
            params = {column: transformation}
            transformer = QuantileTransformation(params)
            score = evaluate_model(x, y, model, scoring, cv, groups, transformer)
            logger.progress(f"   ↪ Tried '{transformation}' → Score: {score:.4f}")

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
