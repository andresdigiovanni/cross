from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class ScaleTransformation:
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

            transformer.fit(x[[column]])
            self.transformers[column] = transformer

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        for column, transformer in self.transformers.items():
            x_transformed[column] = transformer.transform(x[[column]])

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
