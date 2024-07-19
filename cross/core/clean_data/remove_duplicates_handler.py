class RemoveDuplicatesHandler:
    def __init__(self, subset=None, keep=None, config=None):
        self.subset = subset or []
        self.keep = keep or []

        if config:
            self.subset = config.get("subset", [])
            self.keep = config.get("keep", [])

    def get_params(self):
        params = {
            "subset": self.subset,
            "keep": self.keep,
        }
        return params

    def fit(self, df):
        pass

    def transform(self, df):
        if self.keep not in ["first", "last", False]:
            return df

        return df.drop_duplicates(subset=self.subset, keep=self.keep)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
