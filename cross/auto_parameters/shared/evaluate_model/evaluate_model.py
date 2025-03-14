import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def build_pipeline(model, transformer=None):
    steps = []

    # Add the custom transformations
    if transformer:
        steps.append(("transformer", transformer))

    # Impute 0's and select numeric columns
    imputer = SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)
    numeric_transformer = ColumnTransformer(
        [
            ("imputer", imputer, make_column_selector(dtype_include="number")),
        ]
    )
    steps.append(("numeric_processing", numeric_transformer))

    # Add model
    steps.append(("model", model))

    return Pipeline(steps=steps)


def evaluate_model(
    x,
    y,
    model,
    scoring,
    cv=5,
    groups=None,
    transformer=None,
):
    pipe = build_pipeline(model, transformer)
    scores = cross_val_score(
        pipe, x, y, scoring=scoring, cv=cv, groups=groups, n_jobs=-1
    )

    return np.mean(scores)
