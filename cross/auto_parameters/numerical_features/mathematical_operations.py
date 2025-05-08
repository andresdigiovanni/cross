from cross.auto_parameters.shared import ProbeFeatureSelector, RecursiveFeatureAddition
from cross.transformations import MathematicalOperations
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class MathematicalOperationsParamCalculator:
    SYMMETRIC_OPS = ["add", "subtract", "multiply"]
    NON_SYMMETRIC_OPS = ["divide"]

    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        columns = dtypes.numerical_columns(x)
        all_transformations_info = []
        all_selected_features = []

        idxs_columns = range(len(columns))
        total_columns = len(columns)

        logger.task_start("Starting mathematical transformations search")

        # Select operations per columns using probe method
        for i, idx_column_1 in enumerate(idxs_columns, start=1):
            column_1 = columns[idx_column_1]
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column_1}'")

            transformations_info, operations_options = self._generate_operations(
                x, column_1, idx_column_1, columns, idxs_columns
            )
            all_transformations_info.extend(transformations_info)

            transformer = MathematicalOperations(operations_options)
            x_transformed = transformer.fit_transform(x, y)

            selected_features = ProbeFeatureSelector.fit(x_transformed, y, model)
            all_selected_features.extend(selected_features)
            logger.progress(
                f"   ↪ Tried {len(transformations_info)} operations → Selected: {len(selected_features)}"
            )

        selected_transformations = self._select_transformations(
            all_transformations_info, all_selected_features
        )

        # Select final mathematical operations using RFA
        if selected_transformations:
            logger.task_update("Refining selected operations using RFA")
            transformer = MathematicalOperations(selected_transformations)
            x_transformed = transformer.fit_transform(x, y)

            rfa = RecursiveFeatureAddition(model, scoring, direction, cv, groups)
            selected_features = rfa.fit(x_transformed, y)

            selected_transformations = self._select_transformations(
                all_transformations_info, selected_features
            )

        if selected_transformations:
            logger.task_result(
                f"Selected {len(selected_transformations)} mathematical transformation(s)"
            )
            mathematical_operations = MathematicalOperations(selected_transformations)
            return {
                "name": mathematical_operations.__class__.__name__,
                "params": mathematical_operations.get_params(),
            }

        logger.warn("No mathematical transformations was applied to any column")
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

    def _select_transformations(self, transformations_info, selected_features):
        selected_transformations = []

        for transformation in transformations_info:
            if transformation["transformed_column"] in selected_features:
                selected_transformations.append(transformation["operation_option"])

        return selected_transformations
