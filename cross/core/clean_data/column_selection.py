class ColumnSelection:
    def __init__(self, columns=None):
        self.columns = columns or []

    def get_params(self):
        return {"columns": self.columns}

    def fit(self, x, y=None):
        return

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        x_transformed = x_transformed[self.columns]

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
