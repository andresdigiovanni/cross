from tqdm import tqdm

from cross.auto_parameters.shared import ProbeFeatureSelector, RecursiveFeatureAddition
from cross.transformations import MathematicalOperations
from cross.transformations.utils.dtypes import numerical_columns


class MathematicalOperationsParamCalculator:
    SYMMETRIC_OPS = ["add", "subtract", "multiply", "hypotenuse", "mean"]
    NON_SYMMETRIC_OPS = ["divide", "modulus"]

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, verbose
    ):
        columns = numerical_columns(x)
        all_transformations_info = []
        all_selected_features = []

        idxs_columns = range(len(columns))

        # Select operations per columns using probe method
        for idx_column_1 in tqdm(idxs_columns, disable=not verbose):
            column_1 = columns[idx_column_1]

            transformations_info, operations_options = self._generate_operations(
                x, column_1, idx_column_1, columns, idxs_columns
            )
            all_transformations_info.extend(transformations_info)

            transformer = MathematicalOperations(operations_options)
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
            transformer = MathematicalOperations(selected_transformations)
            x_transformed = transformer.fit_transform(x, y)
            new_columns = list(set(x_transformed.columns) - set(x.columns))

            selected_features = RecursiveFeatureAddition.fit(
                x_transformed[new_columns], y, model, scoring, direction, cv, groups
            )

            selected_transformations = self._select_transformations(
                all_transformations_info, selected_features
            )

        if selected_transformations:
            mathematical_operations = MathematicalOperations(selected_transformations)
            return {
                "name": mathematical_operations.__class__.__name__,
                "params": mathematical_operations.get_params(),
            }
        return None

    def _generate_operations(self, x, column_1, idx_column_1, columns, idxs_columns):
        all_transformations_info = []
        all_operations_options = []

        for op in self.SYMMETRIC_OPS:
            transformations_info, operations_options = self._perform_operations(
                x, column_1, idx_column_1, columns, idxs_columns, op, symmetric=True
            )
            all_transformations_info.extend(transformations_info)
            all_operations_options.extend(operations_options)

        for op in self.NON_SYMMETRIC_OPS:
            transformations_info, operations_options = self._perform_operations(
                x, column_1, idx_column_1, columns, idxs_columns, op, symmetric=False
            )
            all_transformations_info.extend(transformations_info)
            all_operations_options.extend(operations_options)

        return all_transformations_info, all_operations_options

    def _perform_operations(
        self, x, column_1, idx_column_1, columns, idxs_columns, op, symmetric
    ):
        transformations_info = []
        operations_options = []

        for idx_column_2 in idxs_columns:
            if symmetric and idx_column_1 >= idx_column_2:
                continue

            if not symmetric and idx_column_1 == idx_column_2:
                continue

            column_2 = columns[idx_column_2]
            operation_option = (column_1, column_2, op)
            operations_options.append(operation_option)

            mathematical_operations = MathematicalOperations([operation_option])
            x_math = mathematical_operations.fit_transform(x)
            math_column_name = list(set(x_math.columns) - set(x.columns))[0]

            transformations_info.append(
                {
                    "operation_option": operation_option,
                    "transformed_column": math_column_name,
                }
            )

        return transformations_info, operations_options

    def _select_transformations(self, all_transformations_info, all_selected_features):
        selected_transformations = []

        for transformation_info in all_transformations_info:
            if transformation_info["transformed_column"] in all_selected_features:
                selected_transformations.append(transformation_info["operation_option"])

        return selected_transformations
