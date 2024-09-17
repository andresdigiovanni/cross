class RemoveDuplicatesHandler:
    def __init__(self, subset=None, keep="first"):
        self.subset = subset or []
        self.keep = keep

    def get_params(self):
        return {"subset": self.subset, "keep": self.keep}

    def fit(self, x, y=None):
        return

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        if self.keep in ["first", "last", False]:
            if y_transformed is not None:
                combined = x_transformed.copy()
                combined["__y"] = y_transformed

                combined = combined.drop_duplicates(
                    subset=self.subset, keep=self.keep
                ).reset_index(drop=True)

                x_transformed = combined.drop(columns=["__y"])
                y_transformed = combined["__y"]

            else:
                x_transformed = x_transformed.drop_duplicates(
                    subset=self.subset, keep=self.keep
                ).reset_index(drop=True)

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
