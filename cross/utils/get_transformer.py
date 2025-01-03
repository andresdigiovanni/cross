from cross.transformations import (
    CastColumns,
    CategoricalEncoding,
    ColumnSelection,
    CyclicalFeaturesTransformer,
    DateTimeTransformer,
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
