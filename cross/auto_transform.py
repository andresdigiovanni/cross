import warnings
from collections import defaultdict
from datetime import datetime
from typing import Callable, List, Optional, Union

import numpy as np

import cross.auto_parameters as pc
from cross.transformations import ColumnSelection
from cross.transformations.utils.dtypes import numerical_columns
from cross.utils import get_transformer


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
        direction (str, optional): "maximize" to increase score or "minimize" to decrease. Defaults to "maximize".
        cv (Union[int, Callable], optional): Number of cross-validation folds or a custom cross-validation generator. Defaults to None.
        groups (Optional[np.ndarray], optional): Group labels for cross-validation splitting. Defaults to None.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        List[dict]: A list of applied transformations.
    """

    if verbose:
        print(f"\n[{date_time()}] Starting experiment to find the best transformations")
        print(f"[{date_time()}] Data shape: {X.shape}")
        print(f"[{date_time()}] Model: {model.__class__.__name__}")
        print(f"[{date_time()}] Scoring: {scoring}\n")

    X = X.copy()
    initial_columns = set(X.columns)
    initial_num_columns = numerical_columns(X)
    transformations = []
    tracked_columns = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        additional_columns = defaultdict(list)

        # transformation - subset - key
        transformations_sequence = [
            (pc.MissingValuesIndicatorParamCalculator, None, None),
            (pc.MissingValuesParamCalculator, None, None),
            (pc.OutliersParamCalculator, None, None),
            (pc.NonLinearTransformationParamCalculator, None, None),
            (pc.NormalizationParamCalculator, None, None),
            (pc.QuantileTransformationParamCalculator, None, None),
            (
                pc.SplineTransformationParamCalculator,
                lambda: initial_num_columns,
                "spline",
            ),
            (
                pc.MathematicalOperationsParamCalculator,
                lambda: initial_num_columns,
                "math",
            ),
            (
                pc.NumericalBinningParamCalculator,
                lambda: initial_num_columns + additional_columns["math"],
                "binning",
            ),
            (
                pc.ScaleTransformationParamCalculator,
                lambda: initial_num_columns
                + additional_columns["spline"]
                + additional_columns["math"]
                + additional_columns["binning"],
                None,
            ),
            (pc.DateTimeTransformerParamCalculator, None, "datetime"),
            (
                pc.CyclicalFeaturesTransformerParamCalculator,
                lambda: additional_columns["datetime"]
                if additional_columns["datetime"]
                else None,
                None,
            ),
            (pc.CategoricalEncodingParamCalculator, None, None),
            (pc.ColumnSelectionParamCalculator, None, None),
            (pc.DimensionalityReductionParamCalculator, None, None),
        ]

        for calculator_cls, subset, key in transformations_sequence:
            calculator = calculator_cls()
            subset_columns = subset() if callable(subset) else subset
            X, new_columns = execute_transformation(
                calculator,
                X,
                y,
                model,
                scoring,
                direction,
                cv,
                groups,
                verbose,
                transformations,
                tracked_columns,
                subset_columns,
            )
            final_columns = set(X.columns)

            if key:
                additional_columns[key] = new_columns

    transformations = filter_transformations(
        transformations, tracked_columns, initial_columns, final_columns
    )
    return transformations


def date_time() -> str:
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
    verbose,
    transformations,
    tracked_columns,
    subset=None,
):
    if verbose:
        print(
            f"\n[{date_time()}] Fitting transformation: {calculator.__class__.__name__}"
        )

    initial_columns = set(X.columns)
    X_subset = X.loc[:, subset] if subset else X

    transformation = calculator.calculate_best_params(
        X_subset, y, model, scoring, direction, cv, groups, verbose
    )

    if transformation:
        transformations.append(transformation)
        transformer = get_transformer(
            transformation["name"], transformation["params"] | {"track_columns": True}
        )
        X = transformer.fit_transform(X, y)

        tracked_columns.append(transformer.tracked_columns)

    return X, list(set(X.columns) - initial_columns)


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
