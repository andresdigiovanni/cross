import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def normalize_data(df, transformation_options):
    df_transformed = df.copy()
    transformers = {}

    for column, transformation in transformation_options.items():
        df_transformed, transformer = normalize_column(
            df_transformed, column, transformation
        )
        transformers[column] = transformer

    return df_transformed, transformers


def normalize_column(df, column, transformation):
    transformer = None

    if transformation == "min_max_scaling":
        transformer = MinMaxScaler()
        df[column] = transformer.fit_transform(df[[column]])

    elif transformation == "standardization":
        transformer = StandardScaler()
        df[column] = transformer.fit_transform(df[[column]])

    elif transformation == "robust_scaling":
        transformer = RobustScaler()
        df[column] = transformer.fit_transform(df[[column]])

    elif transformation == "normalization":
        transformer = np.linalg.norm(df[column])
        df[column] = df[column] / transformer

    elif transformation == "max_abs_scaling":
        transformer = MaxAbsScaler()
        df[column] = transformer.fit_transform(df[[column]])

    elif transformation == "quantile_uniform":
        transformer = QuantileTransformer(output_distribution="uniform")
        df[column] = transformer.fit_transform(df[[column]])

    elif transformation == "quantile_normal":
        transformer = QuantileTransformer(output_distribution="normal")
        df[column] = transformer.fit_transform(df[[column]])

    elif transformation == "log":
        df[column] = np.log1p(df[column])

    elif transformation == "exponential":
        df[column] = np.exp(df[column])

    elif transformation == "yeo_johnson":
        transformer = PowerTransformer(method="yeo-johnson")
        df[column] = transformer.fit_transform(df[[column]])

    return df, transformer
