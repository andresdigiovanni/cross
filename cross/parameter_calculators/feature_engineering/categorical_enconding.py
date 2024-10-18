from tqdm import tqdm

from cross.parameter_calculators.shared import evaluate_model
from cross.transformations.feature_engineering import CategoricalEncoding
from cross.transformations.utils.dtypes import categorical_columns


class CategoricalEncodingParamCalculator:
    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        columns = categorical_columns(x)
        encodings = ["label", "dummy", "binary", "target", "count"]

        best_encodings_options = {}

        with tqdm(total=len(columns) * len(encodings), disable=not verbose) as pbar:
            for column in columns:
                best_score = float("-inf") if direction == "maximize" else float("inf")
                best_encoding = None

                num_unique_values = x[column].nunique()

                for encoding in encodings:
                    pbar.update(1)

                    if encoding == "dummy" and num_unique_values > 20:
                        continue

                    encodings_options = {column: encoding}
                    handler = CategoricalEncoding(encodings_options=encodings_options)
                    score = evaluate_model(x, y, model, scoring, handler)

                    if self._is_score_improved(score, best_score, direction):
                        best_score = score
                        best_encoding = encoding

                if best_encoding:
                    best_encodings_options[column] = best_encoding

        if best_encodings_options:
            categorical_encoding = CategoricalEncoding(
                encodings_options=best_encodings_options
            )
            return {
                "name": categorical_encoding.__class__.__name__,
                "params": categorical_encoding.get_params(),
            }

        return None

    def _is_score_improved(self, score, best_score, direction):
        return (direction == "maximize" and score > best_score) or (
            direction == "minimize" and score < best_score
        )
