from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer

from cross.transformations.utils.dtypes import categorical_columns


class MissingValuesHandler(BaseEstimator, TransformerMixin):
    def __init__(self, handling_options=None, n_neighbors=None):
        self.handling_options = handling_options
        self.n_neighbors = n_neighbors

        self._statistics = {}
        self._imputers = {}

    def get_params(self, deep=True):
        return {
            "handling_options": self.handling_options,
            "n_neighbors": self.n_neighbors,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._statistics = {}
        self._imputers = {}

        for column, action in self.handling_options.items():
            if action == "fill_mean":
                self._statistics[column] = X[column].mean()

            elif action == "fill_median":
                self._statistics[column] = X[column].median()

            elif action == "fill_mode":
                self._statistics[column] = X[column].mode()[0]

            elif action == "fill_knn":
                imputer = KNNImputer(n_neighbors=self.n_neighbors.get(column, 5))
                imputer.fit(X[[column]])
                self._imputers[column] = imputer

            elif action == "most_frequent":
                imputer = SimpleImputer(strategy="most_frequent")
                imputer.fit(X[[column]])
                self._imputers[column] = imputer

        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        cat_columns = categorical_columns(X)

        for column, action in self.handling_options.items():
            if action in ["fill_mean", "fill_median", "fill_mode"]:
                X_transformed[column] = X_transformed[column].fillna(
                    self._statistics[column]
                )

            elif action == "fill_0":
                fill_with = "Unknown" if column in cat_columns else 0
                X_transformed[column] = X_transformed[column].fillna(fill_with)

            elif action == "interpolate":
                X_transformed[column] = X_transformed[column].interpolate()

            elif action in ["fill_knn", "most_frequent"]:
                imputer = self._imputers[column]
                X_transformed[column] = imputer.transform(
                    X_transformed[[column]]
                ).flatten()

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
