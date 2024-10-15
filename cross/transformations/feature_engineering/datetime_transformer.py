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
        X_transformed = X.copy()

        for column in self.datetime_columns:
            X_transformed[f"{column}_year"] = X_transformed[column].dt.year
            X_transformed[f"{column}_month"] = X_transformed[column].dt.month
            X_transformed[f"{column}_day"] = X_transformed[column].dt.day
            X_transformed[f"{column}_weekday"] = X_transformed[column].dt.weekday
            X_transformed[f"{column}_hour"] = X_transformed[column].dt.hour
            X_transformed[f"{column}_minute"] = X_transformed[column].dt.minute
            X_transformed[f"{column}_second"] = X_transformed[column].dt.second

        X_transformed = X_transformed.drop(columns=self.datetime_columns)

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
