import numpy as np
from sklearn.preprocessing import PowerTransformer


class NonLinearTransformation:
    def __init__(self, transformation_options=None, transformers=None):
        self.transformation_options = transformation_options or {}
        self.transformers = transformers or {}

    def get_params(self):
        return {
            "transformation_options": self.transformation_options,
            "transformers": self.transformers,
        }

    def fit(self, x, y=None):
        self.transformers = {}

        for column, transformation in self.transformation_options.items():
            if transformation == "yeo_johnson":
                transformer = PowerTransformer(method="yeo-johnson")
                transformer.fit(x[[column]])
                self.transformers[column] = transformer

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        for column, transformation in self.transformation_options.items():
            if transformation == "log":
                x_transformed[column] = np.log1p(x_transformed[column])

            elif transformation == "exponential":
                x_transformed[column] = np.exp(x_transformed[column])

            elif transformation == "yeo_johnson":
                transformer = self.transformers[column]
                x_transformed[column] = transformer.transform(x_transformed[[column]])

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
