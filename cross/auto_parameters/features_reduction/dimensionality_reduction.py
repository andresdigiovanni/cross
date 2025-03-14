from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import DimensionalityReduction


class DimensionalityReductionParamCalculator:
    def calculate_best_params(
        self, X, y, model, scoring, direction, cv, groups, verbose
    ):
        methods = [
            "factor_analysis",
            "ica",
            "isomap",
            "kernel_pca",
            "lda",
            "lle",
            "pca",
            "truncated_svd",
        ]

        n_features = X.shape[1]
        n_classes = y.nunique()

        n_components_ranges = {
            "factor_analysis": (2, min(50, n_features)),
            "ica": (2, min(50, n_features)),
            "isomap": (2, min(50, n_features)),
            "kernel_pca": (2, min(50, n_features)),
            "lda": (2, min(50, n_features, n_classes - 1)),
            "lle": (2, min(50, n_features)),
            "pca": (2, min(50, n_features)),
            "truncated_svd": (2, min(50, n_features)),
        }

        best_method = None
        best_n_components = None
        best_score = evaluate_model(X, y, model, scoring, cv, groups)

        with tqdm(total=len(methods), disable=not verbose) as pbar:
            for method in methods:
                n_components, score = self._binary_search_optimal_components(
                    X,
                    y,
                    method,
                    n_components_ranges[method],
                    model,
                    scoring,
                    direction,
                    cv,
                    groups,
                )
                pbar.update(1)

                if is_score_improved(score, best_score, direction):
                    best_score = score
                    best_method = method
                    best_n_components = n_components

        if best_method and best_n_components:
            dimensionality_reduction = DimensionalityReduction(
                features=list(X.columns),
                method=best_method,
                n_components=best_n_components,
            )
            return {
                "name": dimensionality_reduction.__class__.__name__,
                "params": dimensionality_reduction.get_params(),
            }

        return None

    def _binary_search_optimal_components(
        self, X, y, method, n_range, model, scoring, direction, cv, groups
    ):
        low, high = n_range
        best_n_components = low
        best_score = float("-inf") if direction == "maximize" else float("inf")
        scores = {}

        while low < high:
            mid1 = low + (high - low) // 3
            mid2 = high - (high - low) // 3

            # Evaluate performance for the two midpoints
            if mid1 not in scores:
                handler_1 = DimensionalityReduction(
                    features=list(X.columns), method=method, n_components=mid1
                )
                score_1 = evaluate_model(X, y, model, scoring, cv, groups, handler_1)
                scores[mid1] = score_1

            score_1 = scores[mid1]

            if mid2 not in scores:
                handler_2 = DimensionalityReduction(
                    features=list(X.columns), method=method, n_components=mid2
                )
                score_2 = evaluate_model(X, y, model, scoring, cv, groups, handler_2)
                scores[mid2] = score_2

            score_2 = scores[mid2]

            if is_score_improved(score_1, best_score, direction):
                best_score = score_1
                best_n_components = mid1

            if is_score_improved(score_2, best_score, direction):
                best_score = score_2
                best_n_components = mid2

            # Narrow the search space based on comparisons
            if is_score_improved(score_2, score_1, direction):
                low = mid1 + 1  # Search in the upper half
            else:
                high = mid2 - 1  # Search in the lower half

        return best_n_components, best_score
