from collections import defaultdict
from typing import Any, Dict, Optional

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import CategoricalEncoding
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class CategoricalEncodingParameterSelector:
    def select_best_parameters(
        self, X, y, model, scoring, direction: str, cv, groups, logger: VerboseLogger
    ) -> Optional[Dict[str, Any]]:
        logger.task_start("Beginning search for optimal categorical encodings.")

        candidate_encodings = self._get_candidate_encodings(X)
        total_columns = len(candidate_encodings)
        best_encoding_config = {}

        for index, (column, encodings) in enumerate(
            candidate_encodings.items(), start=1
        ):
            logger.task_update(
                f"[{index}/{total_columns}] Evaluating encodings for column: '{column}'"
            )

            best_score = float("-inf") if direction == "maximize" else float("inf")
            optimal_encoding = None

            for encoding_method in encodings:
                current_encoding = {column: encoding_method}
                transformer = CategoricalEncoding(current_encoding)

                score = evaluate_model(X, y, model, scoring, cv, groups, transformer)
                logger.progress(f"   ↪ Tried '{encoding_method}' → Score: {score:.4f}")

                if is_score_improved(score, best_score, direction):
                    best_score = score
                    optimal_encoding = encoding_method

            if optimal_encoding:
                logger.task_result(
                    f"Selected encoding for '{column}': {optimal_encoding}"
                )
                best_encoding_config[column] = optimal_encoding

        if best_encoding_config:
            logger.task_result(
                f"Encoding strategy selected for {len(best_encoding_config)} column(s)."
            )
            final_encoder = CategoricalEncoding(best_encoding_config)
            return {
                "name": final_encoder.__class__.__name__,
                "params": final_encoder.get_params(),
            }

        logger.warn("No suitable categorical encodings were identified.")
        return None

    def _get_candidate_encodings(self, X) -> Dict[str, list]:
        categorical_columns = dtypes.categorical_columns(X)
        column_category_counts = {col: X[col].nunique() for col in categorical_columns}

        encoding_options = defaultdict(list)
        base_encodings = [
            "binary",
            "catboost",
            "count",
            "hashing",
            "label",
            "loo",
            "target",
            "woe",
        ]

        for column, unique_count in column_category_counts.items():
            encodings = base_encodings.copy()
            if unique_count <= 15:
                encodings.append("dummy")
            encoding_options[column] = encodings

        return encoding_options
