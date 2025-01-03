import warnings
from datetime import datetime

from cross.auto_parameters.clean_data import (
    ColumnSelectionParamCalculator,
    MissingValuesParamCalculator,
    OutliersParamCalculator,
)
from cross.auto_parameters.feature_engineering import (
    CategoricalEncodingParamCalculator,
    CorrelatedSubstringEncoderParamCalculator,
    CyclicalFeaturesTransformerParamCalculator,
    DateTimeTransformerParamCalculator,
    MathematicalOperationsParamCalculator,
    NumericalBinningParamCalculator,
)
from cross.auto_parameters.preprocessing import (
    NonLinearTransformationParamCalculator,
    ScaleTransformationParamCalculator,
)
from cross.utils import get_transformer


def auto_transform(X, y, model, scoring, direction, verbose=True):
    if verbose:
        date_time = _date_time()
        print(f"\n[{date_time}] Starting experiment to find the bests transformations")
        print(f"[{date_time}] Data shape: {X.shape}")
        print(f"[{date_time}] Model: {model.__class__.__name__}")
        print(f"[{date_time}] Scoring: {scoring}\n")

    X = X.copy()

    transformations = []
    calculators = [
        ("MissingValuesHandler", MissingValuesParamCalculator),
        ("OutliersHandler", OutliersParamCalculator),
        ("NonLinearTransformation", NonLinearTransformationParamCalculator),
        ("ScaleTransformation", ScaleTransformationParamCalculator),
        ("CorrelatedSubstringEncoder", CorrelatedSubstringEncoderParamCalculator),
        ("CategoricalEncoding", CategoricalEncodingParamCalculator),
        ("DateTimeTransformer", DateTimeTransformerParamCalculator),
        ("CyclicalFeaturesTransformer", CyclicalFeaturesTransformerParamCalculator),
        ("NumericalBinning", NumericalBinningParamCalculator),
        ("MathematicalOperations", MathematicalOperationsParamCalculator),
        ("ColumnSelection", ColumnSelectionParamCalculator),
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for name, calculator in calculators:
            if verbose:
                print(f"[{_date_time()}] Fitting transformation: {name}")

            calculator = calculator()
            transformation = calculator.calculate_best_params(
                X, y, model, scoring, direction, verbose
            )
            if transformation:
                transformations.append(transformation)
                name = transformation["name"]
                params = transformation["params"]
                transformer = get_transformer(name, params)
                X = transformer.fit_transform(X)

    return transformations


def _date_time():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")
