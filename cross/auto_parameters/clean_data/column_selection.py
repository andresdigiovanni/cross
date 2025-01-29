from cross.auto_parameters.shared import RecursiveFeatureAddition
from cross.transformations import ColumnSelection
from cross.transformations.utils.dtypes import numerical_columns


class ColumnSelectionParamCalculator:
    def calculate_best_params(
        self, X, y, model, scoring, direction, cv, groups, verbose
    ):
        numeric_columns = numerical_columns(X)
        X = X[numeric_columns]

        rfa = RecursiveFeatureAddition(model, scoring, direction, cv, groups)
        selected_features = rfa.fit(X, y)

        if verbose:
            print(f"Selected {len(selected_features)} features")

        column_selector = ColumnSelection(selected_features)

        return {
            "name": column_selector.__class__.__name__,
            "params": column_selector.get_params(),
        }
