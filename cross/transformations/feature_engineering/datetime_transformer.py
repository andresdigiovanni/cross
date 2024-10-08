class DateTimeTransformer:
    def __init__(self, datetime_columns=None):
        self.datetime_columns = datetime_columns or []

    def get_params(self):
        return {
            "datetime_columns": self.datetime_columns,
        }

    def fit(self, x, y=None):
        pass

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        for column in self.datetime_columns:
            x_transformed[f"{column}_year"] = x_transformed[column].dt.year
            x_transformed[f"{column}_month"] = x_transformed[column].dt.month
            x_transformed[f"{column}_day"] = x_transformed[column].dt.day
            x_transformed[f"{column}_weekday"] = x_transformed[column].dt.weekday
            x_transformed[f"{column}_hour"] = x_transformed[column].dt.hour
            x_transformed[f"{column}_minute"] = x_transformed[column].dt.minute
            x_transformed[f"{column}_second"] = x_transformed[column].dt.second

        x_transformed = x_transformed.drop(columns=self.datetime_columns)

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
