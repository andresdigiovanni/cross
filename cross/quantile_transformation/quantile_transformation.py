from sklearn.preprocessing import QuantileTransformer


class QuantileTransformation:
    def __init__(self, transformation_options):
        self.transformation_options = transformation_options
        self.transformers = {}

    def fit(self, df):
        self.transformers = {}

        for column, transformation in self.transformation_options.items():
            if transformation in ["uniform", "normal"]:
                transformer = QuantileTransformer(output_distribution=transformation)
                transformer.fit(df[[column]])
                self.transformers[column] = transformer

    def transform(self, df):
        df_transformed = df.copy()

        for column, transformation in self.transformation_options.items():
            if transformation in ["uniform", "normal"]:
                transformer = self.transformers[column]
                df_transformed[column] = transformer.transform(df_transformed[[column]])

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
