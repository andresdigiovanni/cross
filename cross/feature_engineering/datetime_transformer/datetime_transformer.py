class DateTimeTransformer:
    def __init__(self, datetime_columns):
        self.datetime_columns = datetime_columns

    def fit(self, df):
        pass

    def transform(self, df):
        df_transformed = df.copy()

        for column in self.datetime_columns:
            df_transformed[f"{column}_year"] = df_transformed[column].dt.year
            df_transformed[f"{column}_month"] = df_transformed[column].dt.month
            df_transformed[f"{column}_day"] = df_transformed[column].dt.day
            df_transformed[f"{column}_hour"] = df_transformed[column].dt.hour
            df_transformed[f"{column}_minute"] = df_transformed[column].dt.minute
            df_transformed[f"{column}_second"] = df_transformed[column].dt.second

        df_transformed = df_transformed.drop(columns=self.datetime_columns)

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
