import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


def build_pipeline(model, transformer=None):
    steps = []

    if transformer:
        steps.append(("transformer", transformer))

    numeric_transformer = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", make_column_selector(dtype_include="number"))
        ]
    )
    steps.append(("numeric_processing", numeric_transformer))
    steps.append(("model", model))

    return Pipeline(steps=steps)


def evaluate_model(
    x,
    y,
    model,
    scoring,
    transformer=None,
):
    # Build pipeline with optional transformer and model
    pipe = build_pipeline(model, transformer)

    # Perform cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, x, y, scoring=scoring, cv=cv, n_jobs=-1)

    return np.mean(scores)
