class ColumnSelection:
    def __init__(self, columns=None, config=None):
        self.columns = columns or []

        if config:
            self.columns = config.get("columns", [])

    def get_params(self):
        params = {
            "columns": self.columns,
        }
        return params

    def fit(self, df):
        pass

    def transform(self, df):
        df_transformed = df.copy()
        return df_transformed[self.columns]

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
