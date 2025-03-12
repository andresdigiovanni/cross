import warnings
from collections import defaultdict
from datetime import datetime
from typing import Callable, List, Optional, Union

import numpy as np

import cross.auto_parameters as pc
from cross.transformations.utils.dtypes import numerical_columns
from cross.utils import get_transformer


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
        transformer = get_transformer(transformation["name"], transformation["params"])
        X = transformer.fit_transform(X, y)

    return X, list(set(X.columns) - initial_columns)


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
    initial_num_columns = numerical_columns(X)
    transformations = []

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
                subset_columns,
            )

            if key:
                additional_columns[key] = new_columns

    return transformations
