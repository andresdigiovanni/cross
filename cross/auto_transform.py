import warnings
from datetime import datetime
from typing import Callable, List, Optional, Union

import numpy as np

from cross.auto_parameters import (
    CategoricalEncodingParamCalculator,
    ColumnSelectionParamCalculator,
    CyclicalFeaturesTransformerParamCalculator,
    DateTimeTransformerParamCalculator,
    DimensionalityReductionParamCalculator,
    MathematicalOperationsParamCalculator,
    MissingValuesIndicatorParamCalculator,
    MissingValuesParamCalculator,
    NonLinearTransformationParamCalculator,
    NormalizationParamCalculator,
    NumericalBinningParamCalculator,
    OutliersParamCalculator,
    QuantileTransformationParamCalculator,
    ScaleTransformationParamCalculator,
    SplineTransformationParamCalculator,
)
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

    def date_time() -> str:
        return datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def execute_transformation(calculator, X, y, subset=None):
        # print status
        if verbose:
            date_time_str = date_time()
            func_name = calculator.__class__.__name__
            print(f"\n[{date_time_str}] Fitting transformation: {func_name}")

        # calculate best params for transformation
        X_subset = X.loc[:, subset] if subset else X
        transformation = calculator.calculate_best_params(
            X_subset, y, model, scoring, direction, cv, groups, verbose
        )

        # apply transformation
        initial_columns = X.columns

        if transformation:
            transformations.append(transformation)
            transformer = get_transformer(
                transformation["name"], transformation["params"]
            )
            X = transformer.fit_transform(X, y)

        new_columns = list(set(X.columns) - set(initial_columns))
        return X, new_columns

    if verbose:
        date_time_str = date_time()
        print(
            f"\n[{date_time_str}] Starting experiment to find the best transformations"
        )
        print(f"[{date_time_str}] Data shape: {X.shape}")
        print(f"[{date_time_str}] Model: {model.__class__.__name__}")
        print(f"[{date_time_str}] Scoring: {scoring}\n")

    X = X.copy()
    orig_num_columns = numerical_columns(X)

    transformations = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # MissingValuesIndicator
        calculator = MissingValuesIndicatorParamCalculator()
        X, _ = execute_transformation(calculator, X, y)

        # MissingValuesHandler
        calculator = MissingValuesParamCalculator()
        X, _ = execute_transformation(calculator, X, y)

        # OutliersHandler
        calculator = OutliersParamCalculator()
        X, _ = execute_transformation(calculator, X, y)

        # NonLinearTransformation
        calculator = NonLinearTransformationParamCalculator()
        X, _ = execute_transformation(calculator, X, y)

        # Normalization
        calculator = NormalizationParamCalculator()
        X, _ = execute_transformation(calculator, X, y)

        # QuantileTransformation
        calculator = QuantileTransformationParamCalculator()
        X, _ = execute_transformation(calculator, X, y)

        # SplineTransformation - original num columns
        calculator = SplineTransformationParamCalculator()
        X, spline_columns = execute_transformation(
            calculator, X, y, subset=orig_num_columns
        )

        # MathematicalOperations
        calculator = MathematicalOperationsParamCalculator()
        X, math_columns = execute_transformation(
            calculator, X, y, subset=orig_num_columns
        )

        # NumericalBinning - original num + math columns
        calculator = NumericalBinningParamCalculator()
        columns_to_transform = orig_num_columns + math_columns
        X, binning_columns = execute_transformation(
            calculator, X, y, subset=columns_to_transform
        )

        # ScaleTransformation - original num + spline + math + binning columns
        calculator = ScaleTransformationParamCalculator()
        columns_to_transform = (
            orig_num_columns + spline_columns + math_columns + binning_columns
        )
        X, _ = execute_transformation(calculator, X, y, subset=columns_to_transform)

        # DateTimeTransformer
        calculator = DateTimeTransformerParamCalculator()
        X, datetime_columns = execute_transformation(calculator, X, y)

        # CyclicalFeaturesTransformer - only for date time features
        if datetime_columns:
            calculator = CyclicalFeaturesTransformerParamCalculator()
            X, _ = execute_transformation(calculator, X, y, subset=datetime_columns)

        # CategoricalEncoding
        calculator = CategoricalEncodingParamCalculator()
        X, _ = execute_transformation(calculator, X, y)

        # ColumnSelection
        calculator = ColumnSelectionParamCalculator()
        X, _ = execute_transformation(calculator, X, y)

        # DimensionalityReduction
        calculator = DimensionalityReductionParamCalculator()
        X, _ = execute_transformation(calculator, X, y)

    return transformations
