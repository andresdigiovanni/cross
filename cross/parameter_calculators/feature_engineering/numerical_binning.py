from tqdm import tqdm

from cross.parameter_calculators.shared import FeatureSelector
from cross.transformations.feature_engineering import NumericalBinning
from cross.transformations.utils.dtypes import numerical_columns


class NumericalBinningParamCalculator:
    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        columns = numerical_columns(x)

        selected_transformations = []
        strategies = ["uniform", "quantile", "kmeans"]
        all_n_bins = [3, 5, 8, 12, 20]

        for column in tqdm(columns, disable=(not verbose)):
            all_transformations_info = []
            all_binning_options = []

            num_unique_values = x[column].nunique()

            for strategy in strategies:
                for n_bins in all_n_bins:
                    if n_bins >= num_unique_values:
                        continue

                    binning_option = (column, strategy, n_bins)
                    all_binning_options.append(binning_option)

                    # Calculate binned column name
                    numerical_binning = NumericalBinning([binning_option])
                    x_binned = numerical_binning.fit_transform(x)
                    binned_column_name = list(set(x_binned.columns) - set(x.columns))[0]

                    all_transformations_info.append(
                        {
                            "binning_option": binning_option,
                            "transformed_column": binned_column_name,
                        }
                    )

            if len(all_binning_options):
                feature_selector = FeatureSelector()
                selected_features = feature_selector.fit(
                    x,
                    y,
                    model,
                    scoring,
                    direction,
                    transformer=NumericalBinning(all_binning_options),
                )
            else:
                selected_features = []

            for transformation_info in all_transformations_info:
                if transformation_info["transformed_column"] in selected_features:
                    selected_transformations.append(
                        transformation_info["binning_option"]
                    )

        if selected_transformations:
            numerical_binning = NumericalBinning(selected_transformations)

            return {
                "name": numerical_binning.__class__.__name__,
                "params": numerical_binning.get_params(),
            }

        return None
