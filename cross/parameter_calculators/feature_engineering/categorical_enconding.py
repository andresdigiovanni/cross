from cross.parameter_calculators.shared import evaluate_model
from cross.transformations.feature_engineering import CategoricalEncoding
from cross.transformations.utils.dtypes import categorical_columns


class CategoricalEncodingParamCalculator:
    def calculate_best_params(self, x, y=None, problem_type=None):
        columns = categorical_columns(x)

        best_encodings_options = {}
        encodings = ["label", "dummy", "binary", "target", "count"]

        for column in columns:
            num_unique_values = x[column].nunique()

            best_score = -float("inf")
            best_encoding = None

            for encoding in encodings:
                if encoding == "dummy" and num_unique_values > 20:
                    continue

                encodings_options = {column: encoding}
                missing_values_handler = CategoricalEncoding(
                    encodings_options=encodings_options
                )

                score = evaluate_model(x, y, problem_type, missing_values_handler)
                if score > best_score:
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
