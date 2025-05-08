from datetime import datetime
from typing import Callable, List, Optional, Union

import numpy as np

import cross.auto_parameters as pc
from cross.auto_parameters.shared import evaluate_model
from cross.transformations import ColumnSelection
from cross.transformations.utils import dtypes
from cross.utils import get_transformer
from cross.utils.verbose import VerboseLogger


def auto_transform(
    X: np.ndarray,
    y: np.ndarray,
    model,
    scoring: str,
    direction: str = "maximize",
    cv: Union[int, Callable] = None,
    groups: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[dict]:
    """Automatically applies a series of data transformations to improve model performance.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable.
        model: Machine learning model with a fit method.
        scoring (str): Scoring metric for evaluation.
        direction (str, optional): "maximize" or "minimize". Defaults to "maximize".
        cv (Union[int, Callable], optional): Cross-validation strategy. Defaults to None.
        groups (Optional[np.ndarray], optional): Group labels for cross-validation splitting. Defaults to None.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        List[dict]: A list of applied transformations.
    """

    logger = VerboseLogger(verbose)

    logger.header("Starting automated transformation search")
    logger.config(f"Input shape: {X.shape}")
    logger.config(f"Model: {model.__class__.__name__}")
    logger.config(f"Scoring metric: '{scoring}' with direction '{direction}'")

    X = X.copy()
    y = y.copy()
    initial_columns = set(X.columns)
    initial_num_columns = dtypes.numerical_columns(X)
    transformations, tracked_columns = [], []

    exclude_from_selection = set()
    exclude_from_dimred = set()

    def wrapper(
        transformer,
        X,
        y,
        transformations,
        tracked_columns,
        subset=None,
    ):
        X_transformed, new_transformations, new_tracked_columns = (
            execute_transformation(
                transformer,
                X,
                y,
                model,
                scoring,
                direction,
                cv,
                groups,
                logger,
                subset,
            )
        )

        transformations.extend(new_transformations)
        tracked_columns.extend(new_tracked_columns)
        new_columns = set(X_transformed.columns) - set(X.columns)

        return X_transformed, transformations, tracked_columns, new_columns

    # Apply Missing and Outlier handling
    transformer = pc.MissingValuesIndicatorParamCalculator()
    X, transformations, tracked_columns, new_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )
    exclude_from_dimred.update(new_columns)

    transformer = pc.MissingValuesParamCalculator()
    X, transformations, tracked_columns, _ = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    transformer = pc.OutliersParamCalculator()
    X, transformations, tracked_columns, _ = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    # Feature Engineering
    transformer = pc.SplineTransformationParamCalculator()
    X, transformations, tracked_columns, new_columns = wrapper(
        transformer, X, y, transformations, tracked_columns, subset=initial_num_columns
    )
    exclude_from_selection.update(new_columns)
    exclude_from_dimred.update(new_columns)

    transformer = pc.NumericalBinningParamCalculator()
    X, transformations, tracked_columns, new_columns = wrapper(
        transformer, X, y, transformations, tracked_columns, subset=initial_num_columns
    )
    exclude_from_dimred.update(new_columns)

    # Distribution Transformations (choose best)
    transformations_1, transformations_2 = [], []
    tracked_columns_1, tracked_columns_2 = [], []

    ## Option 1: NonLinear + Normalization
    transformer = pc.NonLinearTransformationParamCalculator()
    X_1, transformations_1, tracked_columns_1, _ = wrapper(
        transformer, X, y, transformations_1, tracked_columns_1
    )

    transformer = pc.NormalizationParamCalculator()
    X_1, transformations_1, tracked_columns_1, _ = wrapper(
        transformer, X_1, y, transformations_1, tracked_columns_1
    )

    ## Option 2: Quantile Transformation
    transformer = pc.QuantileTransformationParamCalculator()
    X_2, transformations_2, tracked_columns_2, _ = wrapper(
        transformer, X, y, transformations_2, tracked_columns_2
    )

    ## Choose best transformation approach
    score_1 = evaluate_model(X_1, y, model, scoring, cv, groups)
    score_2 = evaluate_model(X_2, y, model, scoring, cv, groups)

    if score_1 > score_2:
        X = X_1
        transformations.extend(transformations_1)
        tracked_columns.extend(tracked_columns_1)
    else:
        X = X_2
        transformations.extend(transformations_2)
        tracked_columns.extend(tracked_columns_2)

    # Apply Mathematical Operations
    transformer = pc.MathematicalOperationsParamCalculator()
    X, transformations, tracked_columns, _ = wrapper(
        transformer, X, y, transformations, tracked_columns, subset=initial_num_columns
    )

    # Final scaling after all transformations
    transformer = pc.ScaleTransformationParamCalculator()
    X, transformations, tracked_columns, _ = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    # Periodic Features
    transformer = pc.DateTimeTransformerParamCalculator()
    X, transformations, tracked_columns, datetime_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    if datetime_columns:
        transformer = pc.CyclicalFeaturesTransformerParamCalculator()
        X, transformations, tracked_columns, new_columns = wrapper(
            transformer,
            X,
            y,
            transformations,
            tracked_columns,
            subset=list(datetime_columns),
        )
        exclude_from_dimred.update(new_columns)

    # Categorical Encoding
    transformer = pc.CategoricalEncodingParamCalculator()
    X, transformations, tracked_columns, new_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )
    exclude_from_selection.update(new_columns)
    exclude_from_dimred.update(new_columns)

    # Dimensionality Reduction
    candidate_columns = dtypes.numerical_columns(X)
    columns_for_selection = [
        col for col in candidate_columns if col not in exclude_from_selection
    ]

    transformer = pc.ColumnSelectionParamCalculator()
    X, transformations, tracked_columns, _ = wrapper(
        transformer,
        X,
        y,
        transformations,
        tracked_columns,
        subset=columns_for_selection,
    )

    candidate_columns = dtypes.numerical_columns(X)
    columns_for_dimred = [
        col for col in candidate_columns if col not in exclude_from_dimred
    ]

    transformer = pc.DimensionalityReductionParamCalculator()
    X, transformations, tracked_columns, _ = wrapper(
        transformer,
        X,
        y,
        transformations,
        tracked_columns,
        subset=columns_for_dimred,
    )

    # Remove unnecessary tranformations
    final_columns = set(X.columns)
    return filter_transformations(
        transformations, tracked_columns, initial_columns, final_columns
    )


def date_time() -> str:
    """Returns the current timestamp as a formatted string."""
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def execute_transformation(
    calculator,
    X,
    y,
    model,
    scoring,
    direction,
    cv,
    groups,
    logger,
    subset=None,
):
    """Executes a given transformation and returns the transformed data along with metadata."""
    X_subset = X.loc[:, subset] if subset else X

    transformation = calculator.calculate_best_params(
        X_subset, y, model, scoring, direction, cv, groups, logger
    )
    if not transformation:
        return X, [], []

    transformer = get_transformer(
        transformation["name"], {**transformation["params"], "track_columns": True}
    )
    X_transformed = transformer.fit_transform(X, y)

    return X_transformed, [transformation], [transformer.tracked_columns]


def filter_transformations(
    transformations, column_dependencies, initial_columns, target_columns
):
    filtered_transformations = []
    required_columns = set(target_columns)

    for index in range(len(transformations) - 1, -1, -1):
        transformation = transformations[index]
        transformation_name = transformation["name"]
        transformation_params = transformation["params"].copy()

        dependency_mapping = column_dependencies[index]
        additional_required_columns = {
            source_col
            for output_col, source_cols in dependency_mapping.items()
            if output_col in required_columns
            for source_col in source_cols
        }
        required_columns.update(additional_required_columns)

        modified_param_key = None

        if "features" in transformation_params:
            transformation_params["features"] = [
                col
                for col in transformation_params["features"]
                if col in required_columns
            ]
            modified_param_key = "features"

        elif "transformation_options" in transformation_params:
            transformation_params["transformation_options"] = {
                key: value
                for key, value in transformation_params[
                    "transformation_options"
                ].items()
                if key in required_columns
            }
            modified_param_key = "transformation_options"

        elif "operations_options" in transformation_params:
            transformation_params["operations_options"] = [
                (col1, col2, op)
                for col1, col2, op in transformation_params["operations_options"]
                if col1 in required_columns and col2 in required_columns
            ]
            modified_param_key = "operations_options"

        if modified_param_key and transformation_params[modified_param_key]:
            filtered_transformations.append(
                {
                    "name": transformation_name,
                    "params": transformation_params,
                }
            )

    # Add column selector to minimize initial columns
    selected_columns = [col for col in initial_columns if col in required_columns]
    column_selector = ColumnSelection(selected_columns)
    selector_transformation = {
        "name": column_selector.__class__.__name__,
        "params": column_selector.get_params(),
    }
    filtered_transformations.append(selector_transformation)

    return filtered_transformations[::-1]
