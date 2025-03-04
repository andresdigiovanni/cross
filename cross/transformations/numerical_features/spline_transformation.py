from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer


class SplineTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_options=None):
        self.transformation_options = transformation_options or {}
        self._transformers = {}

    def get_params(self, deep=True):
        return {
            "transformation_options": self.transformation_options,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._transformers = {}

        for column, options in self.transformation_options.items():
            transformer = SplineTransformer(**options)
            transformer.fit(X[[column]])
            self._transformers[column] = transformer

        return self

    def transform(self, X, y=None):
        X = X.copy()

        for column, transformer in self._transformers.items():
            transformed_values = transformer.transform(X[[column]])
            num_features = transformed_values.shape[1]

            for i in range(num_features):
                X[f"{column}__spline_{i}"] = transformed_values[:, i]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
