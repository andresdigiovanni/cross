from collections import defaultdict

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import CategoricalEncoding
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class CategoricalEncodingParamCalculator:
    def calculate_best_params(
        self, X, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        best_transformation_options = {}
        cat_encodings = self._select_categorical_encodings(X)
        total_columns = len(cat_encodings)

        logger.task_start("Starting categorical encoding search")

        for i, (column, encodings) in enumerate(cat_encodings.items(), start=1):
            logger.task_update(
                f"[{i}/{total_columns}] Evaluating encodings for column: '{column}'"
            )
            best_score = float("-inf") if direction == "maximize" else float("inf")
            best_encoding = None

            for encoding in encodings:
                transformation_options = {column: encoding}
                handler = CategoricalEncoding(transformation_options)

                score = evaluate_model(X, y, model, scoring, cv, groups, handler)
                logger.progress(f"   ↪ Tried '{encoding}' → Score: {score:.4f}")

                if is_score_improved(score, best_score, direction):
                    best_score = score
                    best_encoding = encoding

            if best_encoding:
                logger.task_result(f"Selected encoding for '{column}': {best_encoding}")
                best_transformation_options[column] = best_encoding

        if best_transformation_options:
            logger.task_result(
                f"Encoding applied to {len(best_transformation_options)} column(s)"
            )
            categorical_encoding = CategoricalEncoding(best_transformation_options)
            return {
                "name": categorical_encoding.__class__.__name__,
                "params": categorical_encoding.get_params(),
            }

        logger.warn("No categorical encodings selected for any column")
        return None

    def _select_categorical_encodings(self, X):
        cat_columns = dtypes.categorical_columns(X)
        category_counts = {col: X[col].nunique() for col in cat_columns}

        selected_encodings = defaultdict(list)

        for col, count in category_counts.items():
            encodings = [
                "binary",
                "catboost",
                "count",
                "hashing",
                "label",
                "loo",
                "target",
                "woe",
            ]
            if count <= 15:
                encodings.append("dummy")

            selected_encodings[col] = encodings

        return selected_encodings
