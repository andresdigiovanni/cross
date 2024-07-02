class MissingValuesHandler:
    def __init__(self, handling_options):
        self.handling_options = handling_options
        self.statistics_ = {}

    def fit(self, df):
        self.statistics_ = {}
        for column, action in self.handling_options.items():
            if action == "fill_mean":
                self.statistics_[column] = df[column].mean()

            elif action == "fill_median":
                self.statistics_[column] = df[column].median()

            elif action == "fill_mode":
                self.statistics_[column] = df[column].mode()[0]

    def transform(self, df):
        df_transformed = df.copy()

        for column, action in self.handling_options.items():
            if action == "drop":
                df_transformed = df_transformed.dropna(subset=[column])

            elif action in ["fill_mean", "fill_median", "fill_mode"]:
                if column in self.statistics_:
                    df_transformed[column] = df_transformed[column].fillna(
                        self.statistics_[column]
                    )

            elif action == "fill_0":
                df_transformed[column] = df_transformed[column].fillna(0)

            elif action == "interpolate":
                df_transformed[column] = df_transformed[column].interpolate()

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
