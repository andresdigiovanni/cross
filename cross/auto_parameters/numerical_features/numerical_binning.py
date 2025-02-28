from tqdm import tqdm

from cross.auto_parameters.shared import ProbeFeatureSelector, RecursiveFeatureAddition
from cross.transformations import NumericalBinning
from cross.transformations.utils.dtypes import numerical_columns


class NumericalBinningParamCalculator:
    STRATEGIES = ["uniform", "quantile", "kmeans"]
    ALL_N_BINS = [3, 5, 8, 12, 20]

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, verbose
    ):
        columns = numerical_columns(x)
        all_transformations_info = []
        all_selected_features = []

        # Select operations per columns using probe method
        for column in tqdm(columns, disable=not verbose):
            binning_options, transformations_info = self._generate_binning_options(
                x, column
            )
            all_transformations_info.extend(transformations_info)

            transformer = NumericalBinning(binning_options)
            x_transformed = transformer.fit_transform(x, y)
            new_columns = list(set(x_transformed.columns) - set(x.columns))

            selected_features = ProbeFeatureSelector.fit(
                x_transformed[new_columns], y, model
            )
            all_selected_features.extend(selected_features)

        selected_transformations = self._select_transformations(
            all_transformations_info, all_selected_features
        )

        # Select final binnings using RFA
        if selected_transformations:
            transformer = NumericalBinning(selected_transformations)
            x_transformed = transformer.fit_transform(x, y)
            new_columns = list(set(x_transformed.columns) - set(x.columns))

            rfa = RecursiveFeatureAddition(model, scoring, direction, cv, groups)
            selected_features = rfa.fit(x_transformed[new_columns], y)

            selected_transformations = self._select_transformations(
                all_transformations_info, selected_features
            )

        if selected_transformations:
            numerical_binning = NumericalBinning(selected_transformations)
            return {
                "name": numerical_binning.__class__.__name__,
                "params": numerical_binning.get_params(),
            }

        return None

    def _generate_binning_options(self, x, column):
        all_binning_options = []
        all_transformations_info = []

        num_unique_values = x[column].nunique()

        for strategy in self.STRATEGIES:
            for n_bins in self.ALL_N_BINS:
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

        return all_binning_options, all_transformations_info

    def _select_transformations(self, all_transformations_info, all_selected_features):
        selected_transformations = []

        for transformation_info in all_transformations_info:
            if transformation_info["transformed_column"] in all_selected_features:
                selected_transformations.append(transformation_info["binning_option"])

        return selected_transformations
