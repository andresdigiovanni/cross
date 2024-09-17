import pickle

from cross.core.clean_data import (
    ColumnSelection,
    MissingValuesHandler,
    OutliersHandler,
    RemoveDuplicatesHandler,
)
from cross.core.feature_engineering import (
    CategoricalEncoding,
    CyclicalFeaturesTransformer,
    DateTimeTransformer,
    MathematicalOperations,
    NumericalBinning,
)
from cross.core.preprocessing import (
    CastColumns,
    NonLinearTransformation,
    Normalization,
    QuantileTransformation,
    ScaleTransformation,
)


class CrossTransformer:
    def __init__(self, transformations=None):
        self.transformations = transformations or []

    def load_transformations(self, transformations):
        self.transformations = []

        for transformation in transformations:
            transformer = self._get_transformer(
                transformation["name"], transformation["params"]
            )
            self.transformations.append(transformer)

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
