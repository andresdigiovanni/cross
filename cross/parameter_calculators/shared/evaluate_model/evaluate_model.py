import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


def evaluate_model(
    x,
    y,
    model,
    scoring,
    transformer=None,
):
    steps = []

    if transformer:
        steps.append(("t", transformer))

    # Handle numeric processing
    numeric_transformer = ColumnTransformer(
        transformers=[
            ("n", "passthrough", make_column_selector(dtype_include="number"))
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
