from sklearn.preprocessing import KBinsDiscretizer


class NumericalBinning:
    def __init__(self, binning_options=None, num_bins=None, binners=None):
        self.binning_options = binning_options or {}
        self.num_bins = num_bins or {}
        self.binners = binners or {}

    def get_params(self):
        return {
            "binning_options": self.binning_options,
            "num_bins": self.num_bins,
            "binners": self.binners,
        }

    def fit(self, x, y=None):
        self.binners = {}

        x_filled = x.copy()
        x_filled = x_filled.fillna(0)

        for column, strategy in self.binning_options.items():
            if strategy != "none":
                binner = KBinsDiscretizer(
                    n_bins=self.num_bins[column], encode="ordinal", strategy=strategy
                )
                binner.fit(x_filled[[column]])
                self.binners[column] = binner

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        x_transformed = x_transformed.fillna(0)

        for column, binner in self.binners.items():
            new_column = "{}__{}_{}".format(
                column, self.binning_options[column], self.num_bins[column]
            )
            x_transformed[new_column] = binner.transform(
                x_transformed[[column]]
            ).flatten()

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
