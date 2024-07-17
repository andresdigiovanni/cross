import numpy as np


class CyclicalFeaturesTransformer:
    def __init__(self, columns_periods):
        self.columns_periods = columns_periods

    def fit(self, df):
        pass

    def transform(self, df):
        df_transformed = df.copy()

        for column, period in self.columns_periods.items():
            df_transformed[f"{column}_sin"] = np.sin(
                2 * np.pi * df_transformed[column] / period
            )
            df_transformed[f"{column}_cos"] = np.cos(
                2 * np.pi * df_transformed[column] / period
            )

        df_transformed = df_transformed.drop(columns=self.columns_periods.keys())

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
