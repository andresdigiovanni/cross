class RemoveDuplicatesHandler:
    def __init__(self, subset, keep):
        self.subset = subset
        self.keep = keep

    def fit(self, df):
        pass

    def transform(self, df):
        if self.keep not in ["first", "last", False]:
            return df

        return df.drop_duplicates(subset=self.subset, keep=self.keep)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
