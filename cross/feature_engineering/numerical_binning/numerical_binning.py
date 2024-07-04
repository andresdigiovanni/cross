from sklearn.preprocessing import KBinsDiscretizer


class NumericalBinning:
    def __init__(self, binning_options, num_bins):
        self.binning_options = binning_options
        self.num_bins = num_bins
        self.binners = {}

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

        for column, strategy in self.binning_options.items():
            if column in self.binners:
                binner = self.binners[column]
                df_transformed[column] = binner.transform(
                    df_transformed[[column]]
                ).flatten()

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
