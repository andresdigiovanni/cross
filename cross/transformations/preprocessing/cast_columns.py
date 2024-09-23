import pandas as pd


class CastColumns:
    def __init__(self, cast_options=None):
        self.cast_options = cast_options or {}

    def get_params(self):
        return {"cast_options": self.cast_options}

    def fit(self, x, y=None):
        return

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        for column, dtype_to_cast in self.cast_options.items():
            if dtype_to_cast == "bool":
                x_transformed[column] = x_transformed[column].astype(bool)

            elif dtype_to_cast == "category":
                x_transformed[column] = x_transformed[column].astype(str)

            elif dtype_to_cast == "datetime":
                x_transformed[column] = pd.to_datetime(x_transformed[column])

            elif dtype_to_cast == "number":
                x_transformed[column] = pd.to_numeric(
                    x_transformed[column], errors="coerce"
                )

            elif dtype_to_cast == "timedelta":
                x_transformed[column] = pd.to_timedelta(x_transformed[column])

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
