from itertools import product

from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import NumericalBinning
from cross.transformations.utils.dtypes import numerical_columns


class NumericalBinningParamCalculator:
    STRATEGIES = ["uniform", "quantile", "kmeans"]
    ALL_N_BINS = [3, 5, 8, 12, 20]

    def calculate_best_params(
        self, X, y, model, scoring, direction, cv, groups, verbose
    ):
        columns = numerical_columns(X)
        best_transformations = {}
        base_score = evaluate_model(X, y, model, scoring, cv, groups)

        combinations = list(product(self.STRATEGIES, self.ALL_N_BINS))

        with tqdm(total=len(columns) * len(combinations), disable=not verbose) as pbar:
            for column in columns:
                best_score = base_score
                best_transformation = None

                for strategy, n_bins in combinations:
                    pbar.update(1)

                    transformation_options = {column: (strategy, n_bins)}
                    transformer = NumericalBinning(transformation_options)
                    score = evaluate_model(
                        X, y, model, scoring, cv, groups, transformer
                    )

                    if is_score_improved(score, best_score, direction):
                        best_score = score
                        best_transformation = (strategy, n_bins)

                if best_transformation:
                    best_transformations[column] = best_transformation

        if best_transformations:
            transformer = NumericalBinning(best_transformations)
            return {
                "name": transformer.__class__.__name__,
                "params": transformer.get_params(),
            }

        return None
