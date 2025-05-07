from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer

from cross.transformations.utils import dtypes


class MissingValuesHandler(BaseEstimator, TransformerMixin):
    def __init__(
        self, transformation_options=None, n_neighbors=None, track_columns=False
    ):
        self.transformation_options = transformation_options
        self.n_neighbors = n_neighbors
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._statistics = {}
        self._imputers = {}

    def get_params(self, deep=True):
        return {
            "transformation_options": self.transformation_options,
            "n_neighbors": self.n_neighbors,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._statistics = {}
        self._imputers = {}

        for column, action in self.transformation_options.items():
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
        X = X.copy()
        cat_columns = dtypes.categorical_columns(X)

        for column, action in self.transformation_options.items():
            if action in ["fill_mean", "fill_median", "fill_mode"]:
                X[column] = X[column].fillna(self._statistics[column])

            elif action == "fill_0":
                fill_with = "Unknown" if column in cat_columns else 0
                X[column] = X[column].fillna(fill_with)

            elif action in ["fill_knn", "most_frequent"]:
                imputer = self._imputers[column]
                X[column] = imputer.transform(X[[column]]).flatten()

            if self.track_columns:
                self.tracked_columns[column] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
