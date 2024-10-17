import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from cross.transformations.utils.dtypes import categorical_columns


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_idx):
        self.columns_idx = columns_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.columns_idx]

        elif isinstance(X, np.ndarray):
            return X[:, self.columns_idx]

        else:
            raise TypeError("The data type {} is not compatible.".format(type(X)))


def evaluate_model(
    x,
    y,
    model,
    scoring,
    transformer=None,
    columns_idx=None,
):
    steps = []

    if transformer:
        steps.append(("t", transformer))

    # Add column selection step if columns_idx is provided
    if columns_idx is not None:
        steps.append(("column_selection", ColumnSelector(columns_idx=columns_idx)))

    # Handle categorical encoding
    cat_columns = categorical_columns(x)
    if len(cat_columns):
        encoder = BinaryEncoder()
        steps.append(("e", encoder))

    # Handle numeric processing
    numeric_transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", make_column_selector(dtype_include="number"))
        ]
    )
    steps.append(("numeric_processing", numeric_transformer))

    # Add the model to the pipeline
    steps.append(("m", model))

    # Create pipeline with all steps
    pipe = Pipeline(steps=steps)

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, x, y, scoring=scoring, cv=cv, n_jobs=-1)

    return np.mean(scores)
