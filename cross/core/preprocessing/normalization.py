from sklearn.preprocessing import Normalizer


class Normalization:
    def __init__(self, transformation_options):
        self.transformation_options = transformation_options
        self.transformers = {}

    def fit(self, df):
        self.transformers = {}

        for column, transformation in self.transformation_options.items():
            if transformation == "l1":
                transformer = Normalizer(norm="l1")
                transformer.fit(df[[column]])
                self.transformers[column] = transformer

            elif transformation == "l2":
                transformer = Normalizer(norm="l2")
                transformer.fit(df[[column]])
                self.transformers[column] = transformer

    def transform(self, df):
        df_transformed = df.copy()

        for column, transformation in self.transformation_options.items():
            if transformation in ["l1", "l2"]:
                transformer = self.transformers[column]
                df_transformed[column] = transformer.transform(df_transformed[[column]])

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
