from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import Normalization
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class NormalizationParamCalculator:
    NORMALIZATION_OPTIONS = ["l1", "l2"]

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

        logger.task_start("Starting normalization parameter search")
        base_score = evaluate_model(x, y, model, scoring, cv, groups)
        logger.baseline(f"Base score: {base_score:.4f}")

        for i, column in enumerate(columns, start=1):
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column}'")

            best_params = self._find_best_normalization_for_column(
                x, y, model, scoring, base_score, column, direction, cv, groups, logger
            )

            if best_params:
                transformation_options.update(best_params)
                logger.task_result(
                    f"Selected normalization for '{column}': {list(best_params.values())[0]}"
                )

        if transformation_options:
            logger.task_result(
                f"Normalization applied to {len(transformation_options)} column(s)"
            )
            return self._build_transformation_result(transformation_options)

        logger.warn("No normalization was applied to any column")
        return None

    def _find_best_normalization_for_column(
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

        for norm in self.NORMALIZATION_OPTIONS:
            params = {column: norm}
            transformer = Normalization(params)
            score = evaluate_model(x, y, model, scoring, cv, groups, transformer)
            logger.progress(f"   ↪ Tried '{norm}' → Score: {score:.4f}")

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_params = params

        return best_params

    def _build_transformation_result(self, transformation_options):
        normalization = Normalization(transformation_options=transformation_options)
        return {
            "name": normalization.__class__.__name__,
            "params": normalization.get_params(),
        }
