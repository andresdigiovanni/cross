from cross.transformations import (
    CastColumns,
    CategoricalEncoding,
    ColumnSelection,
    CyclicalFeaturesTransformer,
    DateTimeTransformer,
    DimensionalityReduction,
    MathematicalOperations,
    MissingValuesHandler,
    NonLinearTransformation,
    Normalization,
    NumericalBinning,
    OutliersHandler,
    QuantileTransformation,
    ScaleTransformation,
)


def get_transformer(name, params):
    transformer_mapping = {
        "CastColumns": CastColumns,
        "CategoricalEncoding": CategoricalEncoding,
        "ColumnSelection": ColumnSelection,
        "CyclicalFeaturesTransformer": CyclicalFeaturesTransformer,
        "DateTimeTransformer": DateTimeTransformer,
        "DimensionalityReduction": DimensionalityReduction,
        "MathematicalOperations": MathematicalOperations,
        "MissingValuesHandler": MissingValuesHandler,
        "NonLinearTransformation": NonLinearTransformation,
        "Normalization": Normalization,
        "NumericalBinning": NumericalBinning,
        "OutliersHandler": OutliersHandler,
        "QuantileTransformation": QuantileTransformation,
        "ScaleTransformation": ScaleTransformation,
    }

    if name in transformer_mapping:
        return transformer_mapping[name](**params)

    raise ValueError(f"Unknown transformer: {name}")
