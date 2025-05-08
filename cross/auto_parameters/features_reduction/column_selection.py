from cross.auto_parameters.shared import RecursiveFeatureAddition
from cross.transformations import ColumnSelection
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class ColumnSelectionParamCalculator:
    def calculate_best_params(
        self, X, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        """Calculates the best parameters for column selection."""
        # Filter only numeric columns
        numeric_columns = dtypes.numerical_columns(X)
        X = X[numeric_columns]

        # Start the Recursive Feature Addition process
        logger.task_start("Starting feature selection")

        # Use RecursiveFeatureAddition to select the best features
        rfa = RecursiveFeatureAddition(model, scoring, direction, cv, groups)
        selected_features = rfa.fit(X, y)

        # Print the final result of the selected features
        logger.task_result(f"Selected {len(selected_features)} features")

        # Create the ColumnSelection with the selected features
        column_selector = ColumnSelection(selected_features)

        # Return the name and parameters of the ColumnSelector
        return {
            "name": column_selector.__class__.__name__,
            "params": column_selector.get_params(),
        }
