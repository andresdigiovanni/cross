import json

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from cross import CrossTransformer
from cross.parameter_calculators.shared import evaluate_model


def load_data():
    data = load_iris()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return df


def duplicate_random_rows_and_remove_values(
    df, target_column=None, duplicates_percentage=0.05, missing_percentage=0.05
):
    # Step 1: Randomly duplicate x% rows
    # Select random indices to duplicate
    duplicate_indices = np.random.choice(
        df.index, size=int(len(df) * duplicates_percentage), replace=True
    )

    # Duplicate those rows and add them to the DataFrame
    df_duplicates = df.loc[duplicate_indices]
    df_extended = pd.concat([df, df_duplicates], ignore_index=True)

    # Step 2: Remove x% of the values in each column except the target column
    # Create a mask to mark the values to remove
    mask = np.random.rand(*df_extended.shape) < missing_percentage

    # If a target column is specified, ensure that we do not set NaNs in that column
    if target_column and target_column in df_extended.columns:
        target_index = df_extended.columns.get_loc(target_column)
        mask[:, target_index] = False

    # Apply the mask, setting NaN in the selected locations
    df_extended = df_extended.mask(mask)

    return df_extended


if __name__ == "__main__":
    problem_type = "classification"  # classification or regression

    df = load_data()
    # df = duplicate_random_rows_and_remove_values(df, target_column="target")
    x, y = df.drop(columns="target"), df["target"]

    # Evalute baseline model
    score = evaluate_model(x, y, problem_type)
    print(f"Baseline score: {score}")

    # Auto transformations
    transformer = CrossTransformer()
    transformations = transformer.auto_transform(x, y, problem_type)

    # Evalute model with transformations
    transformer = CrossTransformer(transformations)

    score = evaluate_model(x, y, problem_type, transformer)
    print(f"Transformations score: {score}")

    print(json.dumps(transformations, indent=4))
