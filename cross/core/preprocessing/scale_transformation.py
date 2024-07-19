from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class ScaleTransformation:
    def __init__(self, transformation_options=None, config=None):
        self.transformation_options = transformation_options or {}
        self.transformers = {}

        if config:
            self.transformation_options = config.get("transformation_options", {})
            self.transformers = config.get("transformers", {})

    def get_params(self):
        params = {
            "transformation_options": self.transformation_options,
            "transformers": self.transformers,
        }
        return params

    def fit(self, df):
        self.transformers = {}

        for column, transformation in self.transformation_options.items():
            if transformation == "min_max":
                transformer = MinMaxScaler()

            elif transformation == "standard":
                transformer = StandardScaler()

            elif transformation == "robust":
                transformer = RobustScaler()

            elif transformation == "max_abs":
                transformer = MaxAbsScaler()

            else:
                continue

            transformer.fit(df[[column]])
            self.transformers[column] = transformer

    def transform(self, df):
        df_transformed = df.copy()

        for column, transformer in self.transformers.items():
            df_transformed[column] = transformer.transform(df_transformed[[column]])

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
