from sklearn.impute import KNNImputer, SimpleImputer


class MissingValuesHandler:
    def __init__(self, handling_options, n_neighbors):
        self.handling_options = handling_options
        self.statistics_ = {}
        self.n_neighbors = n_neighbors
        self.imputers = {}

    def fit(self, df):
        self.statistics_ = {}
        self.imputers = {}

        for column, action in self.handling_options.items():
            if action == "fill_mean":
                self.statistics_[column] = df[column].mean()

            elif action == "fill_median":
                self.statistics_[column] = df[column].median()

            elif action == "fill_mode":
                self.statistics_[column] = df[column].mode()[0]

            elif action == "fill_knn":
                imputer = KNNImputer(n_neighbors=self.n_neighbors[column])
                imputer.fit(df[[column]])
                self.imputers[column] = imputer

            elif action == "most_frequent":
                imputer = SimpleImputer(strategy="most_frequent")
                imputer.fit(df[[column]])
                self.imputers[column] = imputer

    def transform(self, df):
        df_transformed = df.copy()

        for column, action in self.handling_options.items():
            if action == "drop":
                df_transformed = df_transformed.dropna(subset=[column])

            elif action in ["fill_mean", "fill_median", "fill_mode"]:
                df_transformed[column] = df_transformed[column].fillna(
                    self.statistics_[column]
                )

            elif action == "fill_0":
                df_transformed[column] = df_transformed[column].fillna(0)

            elif action == "interpolate":
                df_transformed[column] = df_transformed[column].interpolate()

            elif action in ["fill_knn", "most_frequent"]:
                print(f"transform. {action} for {column}")

                imputer = self.imputers[column]
                df_transformed[column] = imputer.transform(
                    df_transformed[[column]]
                ).flatten()

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
