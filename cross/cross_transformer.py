import pickle
import warnings
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin

from cross.parameter_calculators.clean_data import (
    ColumnSelectionParamCalculator,
    MissingValuesParamCalculator,
    OutliersParamCalculator,
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


class CrossTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformations=None):
        self.transformations = transformations

        if isinstance(transformations, list):
            if all(isinstance(t, dict) for t in transformations):
                self.transformations = self._initialize_transformations(transformations)

    def get_params(self, deep=True):
        return {"transformations": self.transformations}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

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
        transformer_mapping = {
            "CategoricalEncoding": CategoricalEncoding,
            "CastColumns": CastColumns,
            "ColumnSelection": ColumnSelection,
            "CyclicalFeaturesTransformer": CyclicalFeaturesTransformer,
            "DateTimeTransformer": DateTimeTransformer,
            "OutliersHandler": OutliersHandler,
            "MathematicalOperations": MathematicalOperations,
            "MissingValuesHandler": MissingValuesHandler,
            "NonLinearTransformation": NonLinearTransformation,
            "Normalization": Normalization,
            "NumericalBinning": NumericalBinning,
            "QuantileTransformation": QuantileTransformation,
            "ScaleTransformation": ScaleTransformation,
        }

        if name in transformer_mapping:
            return transformer_mapping[name](**params)

        raise ValueError(f"Unknown transformer: {name}")

    def save_transformations(self, file_path):
        transformations_data = [
            {"name": type(t).__name__, "params": t.get_params()}
            for t in self.transformations
        ]

        with open(file_path, "wb") as f:
            pickle.dump(transformations_data, f)

    def fit(self, X, y=None):
        X_transformed = X.copy()
        for transformer in self.transformations:
            transformer.fit(X_transformed, y)
            X_transformed = transformer.transform(X_transformed)

        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for transformer in self.transformations:
            X_transformed = transformer.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X, y=None):
        X_transformed = X.copy()
        for transformer in self.transformations:
            X_transformed = transformer.fit_transform(X_transformed, y)

        return X_transformed

    def auto_transform(self, X, y, model, scoring, direction, verbose=True):
        if verbose:
            date_time = self._date_time()
            print(
                f"\n[{date_time}] Starting experiment to find the bests transformations"
            )
            print(f"[{date_time}] Data shape: {X.shape}")
            print(f"[{date_time}] Model: {model.__class__.__name__}")
            print(f"[{date_time}] Scoring: {scoring}\n")

        X_transformed = X.copy()

        transformations = []
        calculators = [
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
                    X_transformed, y, model, scoring, direction, verbose
                )
                if transformation:
                    transformations.append(transformation)
                    name = transformation["name"]
                    params = transformation["params"]
                    transformer = self._get_transformer(name, params)
                    X_transformed = transformer.fit_transform(X_transformed)

        return transformations

    def _date_time(self):
        now = datetime.now()
        return now.strftime("%d/%m/%Y %H:%M:%S")
