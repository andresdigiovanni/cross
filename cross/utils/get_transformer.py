from cross.transformations.clean_data import (
    ColumnSelection,
    MissingValuesHandler,
    OutliersHandler,
)
from cross.transformations.feature_engineering import (
    CategoricalEncoding,
    CorrelatedSubstringEncoder,
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


def get_transformer(name, params):
    transformer_mapping = {
        "CategoricalEncoding": CategoricalEncoding,
        "CastColumns": CastColumns,
        "ColumnSelection": ColumnSelection,
        "CorrelatedSubstringEncoder": CorrelatedSubstringEncoder,
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
