import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class NonLinearTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_options=None):
        self.transformation_options = transformation_options

        self._transformers = {}

    def get_params(self, deep=True):
        return {"transformation_options": self.transformation_options}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._transformers = {}

        for column, transformation in self.transformation_options.items():
            if transformation == "yeo_johnson":
                transformer = PowerTransformer(method="yeo-johnson", standardize=False)
                transformer.fit(X[[column]])
                self._transformers[column] = transformer

        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()

        for column, transformation in self.transformation_options.items():
            if transformation == "log":
                X_transformed[column] = np.log1p(X_transformed[column])

            elif transformation == "exponential":
                X_transformed[column] = np.exp(X_transformed[column])

            elif transformation == "yeo_johnson":
                transformer = self._transformers[column]
                X_transformed[column] = transformer.transform(X_transformed[[column]])

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
