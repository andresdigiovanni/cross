from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin

from cross.utils import get_transformer


class CrossTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformations=None):
        self.transformations = transformations

        if isinstance(transformations, list):
            if all(isinstance(t, dict) for t in transformations):
                self.transformations = self._initialize_transformations(transformations)

    def get_params(self, deep=True):
        return {"transformations": self.transformations}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def _initialize_transformations(self, transformations):
        initialized_transformers = []
        for transformation in transformations:
            transformer = get_transformer(
                transformation["name"], transformation["params"]
            )
            initialized_transformers.append(transformer)
        return initialized_transformers

    def fit(self, X, y=None):
        X = X.copy()
        for transformer in self.transformations:
            transformer.fit(X, y)
            X = transformer.transform(X)

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for transformer in self.transformations:
            X = transformer.transform(X)

        return X

    def fit_transform(self, X, y=None):
        X = X.copy()
        for transformer in self.transformations:
            X = transformer.fit_transform(X, y)

        return X

    def _date_time(self):
        now = datetime.now()
        return now.strftime("%d/%m/%Y %H:%M:%S")
