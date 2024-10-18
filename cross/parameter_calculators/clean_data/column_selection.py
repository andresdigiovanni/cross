from cross.parameter_calculators.shared import FeatureSelector
from cross.transformations.clean_data import ColumnSelection


class ColumnSelectionParamCalculator:
    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        feature_selector = FeatureSelector()
        selected_features = feature_selector.fit(x, y, model, scoring, direction)

        if verbose:
            print(f"Selected {len(selected_features)} features")

        column_selector = ColumnSelection(selected_features)

        return {
            "name": column_selector.__class__.__name__,
            "params": column_selector.get_params(),
        }
