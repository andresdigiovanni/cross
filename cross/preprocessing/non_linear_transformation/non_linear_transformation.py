import numpy as np
from sklearn.preprocessing import PowerTransformer


class NonLinearTransformation:
    def __init__(self, transformation_options):
        self.transformation_options = transformation_options
        self.transformers = {}

    def fit(self, df):
        self.transformers = {}

        for column, transformation in self.transformation_options.items():
            if transformation == "yeo_johnson":
                transformer = PowerTransformer(method="yeo-johnson")
                transformer.fit(df[[column]])
                self.transformers[column] = transformer

    def transform(self, df):
        df_transformed = df.copy()

        for column, transformation in self.transformation_options.items():
            if transformation == "log":
                df_transformed[column] = np.log1p(df_transformed[column])

            elif transformation == "exponential":
                df_transformed[column] = np.exp(df_transformed[column])

            elif transformation == "yeo_johnson":
                transformer = self.transformers[column]
                df_transformed[column] = transformer.transform(df_transformed[[column]])

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
