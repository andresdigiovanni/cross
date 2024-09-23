from sklearn.impute import KNNImputer, SimpleImputer


class MissingValuesHandler:
    def __init__(
        self, handling_options=None, n_neighbors=None, statistics=None, imputers=None
    ):
        self.handling_options = handling_options or {}
        self.n_neighbors = n_neighbors or {}
        self.statistics = statistics or {}
        self.imputers = imputers or {}

    def get_params(self):
        return {
            "handling_options": self.handling_options,
            "n_neighbors": self.n_neighbors,
            "statistics": self.statistics,
            "imputers": self.imputers,
        }

    def fit(self, x, y=None):
        self.statistics = {}
        self.imputers = {}

        for column, action in self.handling_options.items():
            if action == "fill_mean":
                self.statistics[column] = x[column].mean()

            elif action == "fill_median":
                self.statistics[column] = x[column].median()

            elif action == "fill_mode":
                self.statistics[column] = x[column].mode()[0]

            elif action == "fill_knn":
                imputer = KNNImputer(n_neighbors=self.n_neighbors[column])
                imputer.fit(x[[column]])
                self.imputers[column] = imputer

            elif action == "most_frequent":
                imputer = SimpleImputer(strategy="most_frequent")
                imputer.fit(x[[column]])
                self.imputers[column] = imputer

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        for column, action in self.handling_options.items():
            if action == "drop":
                if y_transformed is not None:
                    combined = x_transformed.copy()
                    combined["__y"] = y_transformed

                    combined = combined.dropna(subset=[column])

                    x_transformed = combined.drop(columns=["__y"])
                    y_transformed = combined["__y"]

                else:
                    x_transformed = x_transformed.dropna(subset=[column])

            elif action in ["fill_mean", "fill_median", "fill_mode"]:
                x_transformed[column] = x_transformed[column].fillna(
                    self.statistics[column]
                )

            elif action == "fill_0":
                x_transformed[column] = x_transformed[column].fillna(0)

            elif action == "interpolate":
                x_transformed[column] = x_transformed[column].interpolate()

            elif action in ["fill_knn", "most_frequent"]:
                imputer = self.imputers[column]
                x_transformed[column] = imputer.transform(
                    x_transformed[[column]]
                ).flatten()

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
