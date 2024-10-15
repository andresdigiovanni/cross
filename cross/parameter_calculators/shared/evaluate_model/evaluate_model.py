import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from cross.transformations.utils.dtypes import numerical_columns


def evaluate_model(x, y, problem_type, transformer=None, columns_idx=None):
    if problem_type == "classification":
        model = RandomForestClassifier(n_jobs=-1, random_state=42)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring_func = accuracy_score

    elif problem_type == "regression":
        model = RandomForestRegressor(n_jobs=-1, random_state=42)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring_func = mean_squared_error

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    scores = []

    for train_idx, test_idx in kfold.split(x, y):
        x_train = _select_data(x, train_idx)
        y_train = _select_data(y, train_idx)
        x_test = _select_data(x, test_idx)
        y_test = _select_data(y, test_idx)

        if transformer:
            x_train = transformer.fit_transform(x_train, y_train)
            x_test = transformer.transform(x_test, y_test)

        if columns_idx is not None:
            x_train = x_train.iloc[:, columns_idx]
            x_test = x_test.iloc[:, columns_idx]

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


def _select_data(x, idx):
    if isinstance(x, np.ndarray):
        return x[idx]

    elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.iloc[idx]

    else:
        raise TypeError("The data type {} is not compatible.".format(type(x)))
