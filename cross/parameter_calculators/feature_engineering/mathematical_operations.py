from tqdm import tqdm

from cross.parameter_calculators.shared import FeatureSelector
from cross.transformations.feature_engineering import MathematicalOperations
from cross.transformations.utils.dtypes import numerical_columns


class MathematicalOperationsParamCalculator:
    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        columns = numerical_columns(x)

        symmetric_ops = ["add", "subtract", "multiply", "hypotenuse", "mean"]
        non_symmetric_ops = ["divide", "modulus"]

        all_transformations_info = []
        all_selected_features = []

        idxs_columns = range(len(columns))

        for idx_column_1 in tqdm(idxs_columns, disable=(not verbose)):
            column_1 = columns[idx_column_1]
            all_operations_options = []

            for op in symmetric_ops:
                for idx_column_2 in range(idx_column_1 + 1, len(columns)):
                    column_2 = columns[idx_column_2]

                    operation_option = (column_1, column_2, op)
                    mathematical_operations = MathematicalOperations([operation_option])
                    x_math = mathematical_operations.fit_transform(x)
                    math_column_name = list(set(x_math.columns) - set(x.columns))[0]

                    all_operations_options.append(operation_option)
                    all_transformations_info.append(
                        {
                            "operation_option": operation_option,
                            "transformed_column": math_column_name,
                        }
                    )

            for op in non_symmetric_ops:
                for idx_column_2 in idxs_columns:
                    if idx_column_1 == idx_column_2:
                        continue

                    column_2 = columns[idx_column_2]

                    operation_option = (column_1, column_2, op)
                    mathematical_operations = MathematicalOperations([operation_option])
                    x_math = mathematical_operations.fit_transform(x)
                    math_column_name = list(set(x_math.columns) - set(x.columns))[0]

                    all_operations_options.append(operation_option)
                    all_transformations_info.append(
                        {
                            "operation_option": operation_option,
                            "transformed_column": math_column_name,
                        }
                    )

            feature_selector = FeatureSelector()
            selected_features = feature_selector.fit(
                x,
                y,
                model,
                scoring,
                direction,
                transformer=MathematicalOperations(all_operations_options),
            )
            all_selected_features.extend(selected_features)

        selected_transformations = []
        for transformation_info in all_transformations_info:
            if transformation_info["transformed_column"] in all_selected_features:
                selected_transformations.append(transformation_info["operation_option"])

        # Select most relevants
        feature_selector = FeatureSelector()
        selected_features = feature_selector.fit(
            x,
            y,
            model,
            scoring,
            direction,
            transformer=MathematicalOperations(selected_transformations),
        )

        selected_transformations = []
        for transformation_info in all_transformations_info:
            if transformation_info["transformed_column"] in selected_features:
                selected_transformations.append(transformation_info["operation_option"])

        # Return result
        if selected_transformations:
            mathematical_operations = MathematicalOperations(selected_transformations)

            return {
                "name": mathematical_operations.__class__.__name__,
                "params": mathematical_operations.get_params(),
            }

        return None
