from sklearn.preprocessing import KBinsDiscretizer


class NumericalBinning:
    def __init__(self, binning_options=None, num_bins=None, config=None):
        self.binning_options = binning_options or {}
        self.num_bins = num_bins or {}
        self.binners = {}

        if config:
            self.binning_options = config.get("binning_options", {})
            self.num_bins = config.get("num_bins", {})
            self.binners = config.get("binners", {})

    def get_params(self):
        params = {
            "binning_options": self.binning_options,
            "num_bins": self.num_bins,
            "binners": self.binners,
        }
        return params

    def fit(self, df):
        self.binners = {}

        df_filled = df.copy()
        df_filled = df_filled.fillna(0)

        for column, strategy in self.binning_options.items():
            if strategy != "none":
                binner = KBinsDiscretizer(
                    n_bins=self.num_bins[column], encode="ordinal", strategy=strategy
                )
                binner.fit(df_filled[[column]])
                self.binners[column] = binner

    def transform(self, df):
        df_transformed = df.copy()
        df_transformed = df_transformed.fillna(0)

        for column, binner in self.binners.items():
            new_column = "{}__{}_{}".format(
                column, self.binning_options[column], self.num_bins[column]
            )
            df_transformed[new_column] = binner.transform(
                df_transformed[[column]]
            ).flatten()

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
