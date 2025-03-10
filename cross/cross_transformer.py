from sklearn.base import BaseEstimator, TransformerMixin


class CrossTransformer(BaseEstimator, TransformerMixin):
    def get_params(self, deep=True):
        return {"transformations": self.transformations}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

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
