import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from cross.transformations.utils.dtypes import numerical_columns


def evaluate_model(x, y, problem_type, transformer=None):
    if problem_type in ["binary_classification", "multiclass_classification"]:
        model = RandomForestClassifier()
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring_func = accuracy_score

    elif problem_type == "regression":
        model = RandomForestRegressor()
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring_func = mean_squared_error

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    scores = []

    for train_idx, test_idx in kfold.split(x, y):
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]

        if transformer:
            x_train, y_train = transformer.fit_transform(x_train, y_train)
            x_test, y_test = transformer.transform(x_test, y_test)

        num_columns = numerical_columns(x_train)
        x_train = x_train[num_columns]
        x_test = x_test[num_columns]

        if len(x_train) == 0 or len(x_test) == 0:
            continue

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        if problem_type == "regression":
            score = -scoring_func(y_test, y_pred)
        else:
            score = scoring_func(y_test, y_pred)

        scores.append(score)

    return np.mean(scores) if len(scores) else float("-inf")