import pickle
import warnings
from datetime import datetime

from cross.parameter_calculators.clean_data import (
    ColumnSelectionParamCalculator,
    MissingValuesParamCalculator,
    OutliersParamCalculator,
    RemoveDuplicatesParamCalculator,
)
from cross.parameter_calculators.feature_engineering import (
    CategoricalEncodingParamCalculator,
    CyclicalFeaturesTransformerParamCalculator,
    DateTimeTransformerParamCalculator,
    MathematicalOperationsParamCalculator,
    NumericalBinningParamCalculator,
)
from cross.parameter_calculators.preprocessing import (
    NonLinearTransformationParamCalculator,
    ScaleTransformationParamCalculator,
)
from cross.transformations.clean_data import (
    ColumnSelection,
    MissingValuesHandler,
    OutliersHandler,
    RemoveDuplicatesHandler,
)
from cross.transformations.feature_engineering import (
    CategoricalEncoding,
    CyclicalFeaturesTransformer,
    DateTimeTransformer,
    MathematicalOperations,
    NumericalBinning,
)
from cross.transformations.preprocessing import (
    CastColumns,
    NonLinearTransformation,
    Normalization,
    QuantileTransformation,
    ScaleTransformation,
)


class CrossTransformer:
    def __init__(self, transformations=None):
        self.transformations = []

        if isinstance(transformations, list):
            if all(isinstance(t, dict) for t in transformations):
                self.transformations = self._initialize_transformations(transformations)
            else:
                self.transformations = transformations

    def load_transformations(self, file_path):
        with open(file_path, "rb") as f:
            transformations = pickle.load(f)

        self.transformations = self._initialize_transformations(transformations)

    def _initialize_transformations(self, transformations):
        initialized_transformers = []
        for transformation in transformations:
            transformer = self._get_transformer(
                transformation["name"], transformation["params"]
            )
            initialized_transformers.append(transformer)
        return initialized_transformers

    def _get_transformer(self, name, params):
        if name == "CategoricalEncoding":
            return CategoricalEncoding(**params)

        if name == "CastColumns":
            return CastColumns(**params)

        if name == "ColumnSelection":
            return ColumnSelection(**params)

        if name == "CyclicalFeaturesTransformer":
            return CyclicalFeaturesTransformer(**params)

        if name == "DateTimeTransformer":
            return DateTimeTransformer(**params)

        if name == "OutliersHandler":
            return OutliersHandler(**params)

        if name == "MathematicalOperations":
            return MathematicalOperations(**params)

        if name == "MissingValuesHandler":
            return MissingValuesHandler(**params)

        if name == "NonLinearTransformation":
            return NonLinearTransformation(**params)

        if name == "Normalization":
            return Normalization(**params)

        if name == "NumericalBinning":
            return NumericalBinning(**params)

        if name == "QuantileTransformation":
            return QuantileTransformation(**params)

        if name == "RemoveDuplicatesHandler":
            return RemoveDuplicatesHandler(**params)

        if name == "ScaleTransformation":
            return ScaleTransformation(**params)

    def save_transformations(self, file_path):
        transformations_data = [
            {"name": type(t).__name__, "params": t.get_params()}
            for t in self.transformations
        ]

        with open(file_path, "wb") as f:
            pickle.dump(transformations_data, f)

    def fit(self, x, y=None):
        x_transformed = x.copy()

        for transformer in self.transformations:
            transformer.fit(x_transformed, y)
            x_transformed = transformer.transform(x_transformed)

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        for transformer in self.transformations:
            if y_transformed is not None:
                x_transformed, y_transformed = transformer.transform(
                    x_transformed, y_transformed
                )

            else:
                x_transformed = transformer.transform(x_transformed)

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        for transformer in self.transformations:
            if y_transformed is not None:
                x_transformed, y_transformed = transformer.fit_transform(
                    x_transformed, y_transformed
                )

            else:
                x_transformed = transformer.fit_transform(x_transformed)

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def auto_transform(self, x, y, problem_type, verbose=True):
        if verbose:
            date_time = self._date_time()
            print(
                f"\n[{date_time}] Starting experiment to find the bests transformations"
            )
            print(f"[{date_time}] Shape: {x.shape}. Problem type: {problem_type}\n")

        x_transformed = x.copy()
        y_transformed = y.copy()

        transformations = []
        calculators = [
            ("RemoveDuplicatesHandler", RemoveDuplicatesParamCalculator),
            ("MissingValuesHandler", MissingValuesParamCalculator),
            ("OutliersHandler", OutliersParamCalculator),
            ("NonLinearTransformation", NonLinearTransformationParamCalculator),
            ("ScaleTransformation", ScaleTransformationParamCalculator),
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
                    print(f"[{self._date_time()}] Fitting transformation: {name}")

                calculator = calculator()
                transformation = calculator.calculate_best_params(
                    x_transformed, y_transformed, problem_type, verbose
                )

                if transformation:
                    transformations.append(transformation)

                    transformer = self._get_transformer(
                        transformation["name"], transformation["params"]
                    )
                    x_transformed, y_transformed = transformer.fit_transform(
                        x_transformed, y_transformed
                    )

        return transformations

    def _date_time(self):
        now = datetime.now()
        return now.strftime("%d/%m/%Y %H:%M:%S")
