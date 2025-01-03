from sklearn.base import BaseEstimator, TransformerMixin


class DateTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_columns=None):
        self.datetime_columns = datetime_columns or []

    def get_params(self, deep=True):
        return {
            "datetime_columns": self.datetime_columns,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        return self  # No fitting necessary, but method required for compatibility

    def transform(self, X, y=None):
        X = X.copy()

        for column in self.datetime_columns:
            X[f"{column}_year"] = X[column].dt.year
            X[f"{column}_month"] = X[column].dt.month
            X[f"{column}_day"] = X[column].dt.day
            X[f"{column}_weekday"] = X[column].dt.weekday
            X[f"{column}_hour"] = X[column].dt.hour
            X[f"{column}_minute"] = X[column].dt.minute
            X[f"{column}_second"] = X[column].dt.second

        X = X.drop(columns=self.datetime_columns)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
