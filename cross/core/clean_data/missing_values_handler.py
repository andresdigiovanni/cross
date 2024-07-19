from sklearn.impute import KNNImputer, SimpleImputer


class MissingValuesHandler:
    def __init__(self, handling_options=None, n_neighbors=None, config=None):
        self.handling_options = handling_options or {}
        self.n_neighbors = n_neighbors or {}
        self.statistics = {}
        self.imputers = {}

        if config:
            self.handling_options = config.get("handling_options", {})
            self.n_neighbors = config.get("n_neighbors", {})
            self.statistics = config.get("statistics", {})
            self.imputers = config.get("imputers", {})

    def get_params(self):
        params = {
            "handling_options": self.handling_options,
            "n_neighbors": self.n_neighbors,
            "statistics": self.statistics,
            "imputers": self.imputers,
        }
        return params

    def fit(self, df):
        self.statistics = {}
        self.imputers = {}

        for column, action in self.handling_options.items():
            if action == "fill_mean":
                self.statistics[column] = df[column].mean()

            elif action == "fill_median":
                self.statistics[column] = df[column].median()

            elif action == "fill_mode":
                self.statistics[column] = df[column].mode()[0]

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
                    self.statistics[column]
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
