from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class ScaleTransformation:
    def __init__(self, transformation_options):
        self.transformation_options = transformation_options
        self.transformers = {}

    def fit(self, df):
        self.transformers = {}

        for column, transformation in self.transformation_options.items():
            if transformation == "min_max":
                transformer = MinMaxScaler()
                transformer.fit(df[[column]])
                self.transformers[column] = transformer

            elif transformation == "standard":
                transformer = StandardScaler()
                transformer.fit(df[[column]])
                self.transformers[column] = transformer

            elif transformation == "robust":
                transformer = RobustScaler()
                transformer.fit(df[[column]])
                self.transformers[column] = transformer

            elif transformation == "max_abs":
                transformer = MaxAbsScaler()
                transformer.fit(df[[column]])
                self.transformers[column] = transformer

    def transform(self, df):
        df_transformed = df.copy()

        for column, transformation in self.transformation_options.items():
            if transformation in ["min_max", "standard", "robust", "max_abs"]:
                transformer = self.transformers[column]
                df_transformed[column] = transformer.transform(df_transformed[[column]])

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
