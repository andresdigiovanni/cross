from datetime import datetime
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

import cross.auto_parameters as pc
from cross.auto_parameters.shared import evaluate_model
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
    distribution_similarity_threshold: float = 0.05,
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
        distribution_similarity_threshold (float, optional): Significance level to accept that distributions are similar.
            Used to select a representative subset of the data. Defaults to 0.05.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        List[dict]: A list of applied transformations.
    """

    if verbose:
        print(f"\n[{date_time()}] Starting transformation search")
        print(f"[{date_time()}] Data shape: {X.shape}")
        print(f"[{date_time()}] Model: {model.__class__.__name__}")
        print(f"[{date_time()}] Scoring: {scoring}\n")

    X = X.copy()
    y = y.copy()
    initial_columns = set(X.columns)
    initial_num_columns = numerical_columns(X)
    transformations, tracked_columns = [], []

    X, y = find_minimal_representative_sample(
        X, y, threshold=distribution_similarity_threshold
    )

    if verbose:
        print(f"[{date_time()}] Resampled data: {X.shape}")

    def wrapper(transformer, X, y, transformations, tracked_columns, subset=None):
        X, new_transformations, new_tracked_columns = execute_transformation(
            transformer,
            X,
            y,
            model,
            scoring,
            direction,
            cv,
            groups,
            verbose,
            subset,
        )

        transformations.extend(new_transformations)
        tracked_columns.extend(new_tracked_columns)

        return X, transformations, tracked_columns

    # Apply Missing and Outlier handling
    transformer = pc.MissingValuesIndicatorParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    transformer = pc.MissingValuesParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    transformer = pc.OutliersParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    # Feature Engineering
    transformer = pc.SplineTransformationParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns, initial_num_columns
    )

    transformer = pc.NumericalBinningParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns, initial_num_columns
    )

    # Distribution Transformations (choose best)
    transformations_1, transformations_2 = [], []
    tracked_columns_1, tracked_columns_2 = [], []

    ## Option 1: NonLinear + Normalization
    transformer = pc.NonLinearTransformationParamCalculator()
    X_1, transformations_1, tracked_columns_1 = wrapper(
        transformer, X, y, transformations_1, tracked_columns_1
    )

    transformer = pc.NormalizationParamCalculator()
    X_1, transformations_1, tracked_columns_1 = wrapper(
        transformer, X_1, y, transformations_1, tracked_columns_1
    )

    ## Option 2: Quantile Transformation
    transformer = pc.QuantileTransformationParamCalculator()
    X_2, transformations_2, tracked_columns_2 = wrapper(
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
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns, initial_num_columns
    )

    # Final scaling after all transformations
    transformer = pc.ScaleTransformationParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    # Periodic Features
    datetime_initial_columns = set(X.columns)
    transformer = pc.DateTimeTransformerParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )
    datetime_columns = set(X.columns) - datetime_initial_columns

    if datetime_columns:
        transformer = pc.CyclicalFeaturesTransformerParamCalculator()
        X, transformations, tracked_columns = wrapper(
            transformer, X, y, transformations, tracked_columns, datetime_columns
        )

    # Categorical Encoding
    transformer = pc.CategoricalEncodingParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    # Dimensionality Reduction
    transformer = pc.ColumnSelectionParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
    )

    transformer = pc.DimensionalityReductionParamCalculator()
    X, transformations, tracked_columns = wrapper(
        transformer, X, y, transformations, tracked_columns
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
    calculator, X, y, model, scoring, direction, cv, groups, verbose, subset=None
):
    """Executes a given transformation and returns the transformed data along with metadata."""
    if verbose:
        print(
            f"\n[{date_time()}] Applying transformation: {calculator.__class__.__name__}"
        )

    X_subset = X.loc[:, subset] if subset else X

    transformation = calculator.calculate_best_params(
        X_subset, y, model, scoring, direction, cv, groups, verbose
    )
    if not transformation:
        return X, [], []

    transformer = get_transformer(
        transformation["name"], {**transformation["params"], "track_columns": True}
    )
    X_transformed = transformer.fit_transform(X, y)

    return X_transformed, [transformation], [transformer.tracked_columns]


def find_minimal_representative_sample(
    X, y, threshold=0.05, step_fraction=0.05, random_state=42
):
    """
    Finds the minimal sample size necessary to maintain the original dataset's distribution.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        threshold (float, optional): Significance level to accept that distributions are similar. Defaults to 0.05.
        step_fraction (float, optional): Percentage of data to add in each iteration (e.g., 0.05 for 5%). Defaults to 0.05.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (X_reduced, y_reduced) where:
            - X_reduced (pd.DataFrame or np.ndarray): Reduced feature matrix.
            - y_reduced (pd.Series or np.ndarray): Reduced target vector.
    """
    np.random.seed(random_state)

    # Ensure X and y are pandas DataFrame/Series
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y = pd.Series(y) if not isinstance(y, pd.Series) else y

    # Initial sample size (step_fraction of the total dataset)
    step_size = max(1, int(len(X) * step_fraction))  # Ensure at least 1 sample
    sample_size = max(100, step_size)

    while sample_size <= len(X):  # Stop when the sample size reaches the full dataset
        # Take a new sample with the current sample size
        sample_data = X.sample(n=sample_size, random_state=random_state)
        sample_labels = y.loc[sample_data.index]

        all_features_match = True

        for col in X.columns:
            if (
                X[col].dtype == "object" or X[col].dtype == "category"
            ):  # Categorical variables
                original_counts = X[col].value_counts()
                sample_counts = sample_data[col].value_counts()

                contingency_table = (
                    pd.concat([original_counts, sample_counts], axis=1).fillna(0).values
                )

                _, p_value, _, _ = chi2_contingency(contingency_table)
            else:  # Numerical variables
                p_value = ks_2samp(X[col].dropna(), sample_data[col].dropna()).pvalue

            if (
                p_value < threshold
            ):  # If distributions are significantly different, increase sample size
                all_features_match = False
                break

        # Check the distribution of the target variable
        original_counts_y = y.value_counts()
        sample_counts_y = sample_labels.value_counts()

        contingency_table_y = (
            pd.concat([original_counts_y, sample_counts_y], axis=1).fillna(0).values
        )
        _, p_value_y, _, _ = chi2_contingency(contingency_table_y)

        if p_value_y < threshold:
            all_features_match = False

        if all_features_match:
            return sample_data, sample_labels  # Return the optimal subset

        sample_size += step_size  # Increase the sample size

    return X, y  # If no optimal subset is found, return the original dataset


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
