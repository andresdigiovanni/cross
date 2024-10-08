from tqdm import tqdm

from cross.parameter_calculators.shared import FeatureSelector
from cross.transformations.feature_engineering import NumericalBinning
from cross.transformations.utils.dtypes import numerical_columns


class NumericalBinningParamCalculator:
    def calculate_best_params(self, x, y, problem_type, verbose):
        columns = numerical_columns(x)

        selected_transformations = []
        strategies = ["uniform", "quantile", "kmeans"]
        all_n_bins = [3, 5, 8, 12, 20]

        for column in tqdm(columns, disable=(not verbose)):
            all_transformations_info = []
            all_binning_options = {}
            all_n_bins_options = {}

            for n_bins in all_n_bins:
                for strategy in strategies:
                    binning_options = {column: strategy}
                    n_bins_options = {column: n_bins}
                    all_binning_options[column] = strategy
                    all_n_bins_options[column] = n_bins

                    numerical_binning = NumericalBinning(
                        binning_options, n_bins_options
                    )
                    x_binned = numerical_binning.fit_transform(x)
                    binned_column_name = list(set(x_binned.columns) - set(x.columns))[0]

                    all_transformations_info.append(
                        {
                            "column": column,
                            "strategy": strategy,
                            "n_bins": n_bins,
                            "transformed_column": binned_column_name,
                        }
                    )

            feature_selector = FeatureSelector()
            _, selected_features = feature_selector.fit(
                x,
                y,
                problem_type,
                maximize=(problem_type == "classification"),
                transformer=NumericalBinning(all_binning_options, all_n_bins_options),
            )

            for transformation_info in all_transformations_info:
                if transformation_info["transformed_column"] in selected_features:
                    selected_transformations.append(
                        {
                            "column": transformation_info["column"],
                            "strategy": transformation_info["strategy"],
                            "n_bins": transformation_info["n_bins"],
                        }
                    )

        if selected_transformations:
            best_binning_options = {}
            best_n_bins_options = {}

            for t in selected_transformations:
                if t["column"] not in best_binning_options:
                    best_binning_options[t["column"]] = t["strategy"]
                    best_n_bins_options[t["column"]] = t["n_bins"]

            numerical_binning = NumericalBinning(
                best_binning_options, best_n_bins_options
            )
            return {
                "name": numerical_binning.__class__.__name__,
                "params": numerical_binning.get_params(),
            }

        return None
