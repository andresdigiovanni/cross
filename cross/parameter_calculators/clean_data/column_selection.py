from cross.parameter_calculators.shared import FeatureSelector
from cross.transformations.clean_data import ColumnSelection


class ColumnSelectionParamCalculator:
    def calculate_best_params(self, x, y, problem_type, verbose):
        feature_selector = FeatureSelector()
        _, selected_features = feature_selector.fit(
            x,
            y,
            problem_type,
            maximize=(problem_type == "classification"),
        )

        column_selector = ColumnSelection(selected_features)

        return {
            "name": column_selector.__class__.__name__,
            "params": column_selector.get_params(),
        }
