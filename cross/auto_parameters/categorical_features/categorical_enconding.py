from tqdm import tqdm

from cross.auto_parameters.shared import evaluate_model
from cross.auto_parameters.shared.utils import is_score_improved
from cross.transformations import CategoricalEncoding
from cross.transformations.utils.dtypes import categorical_columns


class CategoricalEncodingParamCalculator:
    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, verbose
    ):
        columns = categorical_columns(x)
        encodings = [
            "backward_diff",
            "basen",
            "binary",
            "catboost",
            "count",
            "dummy",
            "glmm",
            "gray",
            "hashing",
            "helmert",
            "james_stein",
            "label",
            "loo",
            "m_estimate",
            "onehot",
            # "ordinal",
            "polynomial",
            "quantile",
            "rankhot",
            "sum",
            "target",
            "woe",
        ]

        best_transformation_options = {}

        with tqdm(total=len(columns) * len(encodings), disable=not verbose) as pbar:
            for column in columns:
                best_score = float("-inf") if direction == "maximize" else float("inf")
                best_encoding = None

                for encoding in encodings:
                    pbar.update(1)

                    transformation_options = {column: encoding}
                    handler = CategoricalEncoding(transformation_options)
                    score = evaluate_model(x, y, model, scoring, cv, groups, handler)

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
