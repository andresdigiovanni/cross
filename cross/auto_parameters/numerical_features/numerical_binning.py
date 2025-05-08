from itertools import product

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import NumericalBinning
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class NumericalBinningParamCalculator:
    STRATEGIES = ["uniform", "quantile"]
    ALL_N_BINS = [3, 8, 20]

    def calculate_best_params(
        self, X, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        columns = dtypes.numerical_columns(X)
        total_columns = len(columns)
        best_transformations = {}
        combinations = list(product(self.STRATEGIES, self.ALL_N_BINS))

        logger.task_start("Starting numerical binning search")
        base_score = evaluate_model(X, y, model, scoring, cv, groups)
        logger.baseline(f"Base score: {base_score:.4f}")

        for i, column in enumerate(columns, start=1):
            n_unique = X[column].nunique()
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column}'")

            best_score = base_score
            best_transformation = None

            for strategy, n_bins in combinations:
                if n_unique <= n_bins:
                    continue

                transformation_options = {column: (strategy, n_bins)}
                transformer = NumericalBinning(transformation_options)

                score = evaluate_model(X, y, model, scoring, cv, groups, transformer)
                logger.progress(
                    f"   ↪ Tried '{strategy}' with {n_bins} bins → Score: {score:.4f}"
                )

                if is_score_improved(score, best_score, direction):
                    best_score = score
                    best_transformation = (strategy, n_bins)

            if best_transformation:
                logger.task_result(
                    f"Selected numerical binning for '{column}': {best_transformation[0]} with {best_transformation[1]} bins"
                )
                best_transformations[column] = best_transformation

        if best_transformations:
            logger.task_result(
                f"Numerical binning applied to {len(best_transformations)} column(s)"
            )
            transformer = NumericalBinning(best_transformations)
            return {
                "name": transformer.__class__.__name__,
                "params": transformer.get_params(),
            }

        logger.warn("No numerical binning was applied to any column")
        return None
