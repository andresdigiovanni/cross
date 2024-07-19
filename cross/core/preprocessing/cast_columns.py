import pandas as pd


class CastColumns:
    def __init__(self, cast_options=None, config=None):
        self.cast_options = cast_options or {}

        if config:
            self.cast_options = config.get("cast_options", {})

    def get_params(self):
        params = {
            "cast_options": self.cast_options,
        }
        return params

    def fit(self, df):
        pass

    def transform(self, df):
        df_transformed = df.copy()

        for column, dtype_to_cast in self.cast_options.items():
            if dtype_to_cast == "bool":
                df_transformed[column] = df_transformed[column].astype(bool)

            elif dtype_to_cast == "category":
                df_transformed[column] = df_transformed[column].astype(str)

            elif dtype_to_cast == "datetime":
                df_transformed[column] = pd.to_datetime(df_transformed[column])

            elif dtype_to_cast == "number":
                df_transformed[column] = pd.to_numeric(
                    df_transformed[column], errors="coerce"
                )

            elif dtype_to_cast == "timedelta":
                df_transformed[column] = pd.to_timedelta(df_transformed[column])

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
