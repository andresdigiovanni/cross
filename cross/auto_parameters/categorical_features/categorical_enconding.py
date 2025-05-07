from collections import defaultdict

from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import CategoricalEncoding
from cross.transformations.utils import dtypes


class CategoricalEncodingParamCalculator:
    def calculate_best_params(
        self, X, y, model, scoring, direction, cv, groups, verbose
    ):
        best_transformation_options = {}
        cat_encodings = self._select_categorical_encodings(X)

        for column, encodings in tqdm(cat_encodings.items(), disable=not verbose):
            best_score = float("-inf") if direction == "maximize" else float("inf")
            best_encoding = None

            for encoding in encodings:
                transformation_options = {column: encoding}
                handler = CategoricalEncoding(transformation_options)
                score = evaluate_model(X, y, model, scoring, cv, groups, handler)

                if is_score_improved(score, best_score, direction):
                    best_score = score
                    best_encoding = encoding

            if best_encoding:
                best_transformation_options[column] = best_encoding

        if best_transformation_options:
            categorical_encoding = CategoricalEncoding(best_transformation_options)
            return {
                "name": categorical_encoding.__class__.__name__,
                "params": categorical_encoding.get_params(),
            }

        return None

    def _select_categorical_encodings(self, X):
        cat_columns = dtypes.categorical_columns(X)
        category_counts = {col: X[col].nunique() for col in cat_columns}

        selected_encodings = defaultdict(list)

        for col, count in category_counts.items():
            encodings = [
                "basen",
                "binary",
                "catboost",
                "count",
                "glmm",
                "gray",
                "hashing",
                "james_stein",
                "label",
                "loo",
                "m_estimate",
                "quantile",
                "target",
                "woe",
            ]
            if count <= 15:
                encodings.extend(
                    [
                        "backward_diff",
                        "dummy",
                        "helmert",
                        "onehot",
                        "polynomial",
                        "rankhot",
                        "sum",
                    ]
                )

            selected_encodings[col] = encodings

        return selected_encodings
