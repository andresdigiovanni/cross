import pandas as pd
from category_encoders import BinaryEncoder, TargetEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


class CategoricalEncoding:
    def __init__(self, encodings_options, target_column=None, ordinal_orders=None):
        self.encodings_options = encodings_options
        self.encoders = {}
        self.target_column = target_column
        self.ordinal_orders = ordinal_orders

    def fit(self, df):
        self.encoders = {}

        df_filled = df.copy()
        df_filled = df_filled.fillna("Unknown")

        for column, transformation in self.encodings_options.items():
            if transformation == "label":
                transformer = LabelEncoder()
                transformer.fit(df_filled[column])
                self.encoders[column] = transformer

            elif transformation == "ordinal":
                transformer = OrdinalEncoder(categories=[self.ordinal_orders[column]])
                transformer.fit(df_filled[[column]])
                self.encoders[column] = transformer

            elif transformation == "onehot":
                transformer = OneHotEncoder(sparse_output=False)
                transformer.fit(df_filled[[column]])
                self.encoders[column] = transformer

            elif transformation == "dummy":
                transformer = OneHotEncoder(sparse_output=False, drop="first")
                transformer.fit(df_filled[[column]])
                self.encoders[column] = transformer

            elif transformation == "binary":
                transformer = BinaryEncoder()
                transformer.fit(df_filled[[column]])
                self.encoders[column] = transformer

            elif (
                transformation == "target"
                and self.target_column is not None
                and self.target_column != ""
            ):
                transformer = TargetEncoder()
                transformer.fit(df_filled[[column]], df_filled[self.target_column])
                self.encoders[column] = transformer

            elif transformation == "count":
                self.encoders[column] = df_filled[column].value_counts().to_dict()

    def transform(self, df):
        df_transformed = df.copy()
        df_transformed = df_transformed.fillna("Unknown")

        for column, transformation in self.encodings_options.items():
            if column in self.encoders:
                if transformation in ["label", "ordinal", "target"]:
                    transformer = self.encoders[column]
                    df_transformed[column] = transformer.transform(
                        df_transformed[[column]]
                    )

                elif transformation in ["onehot", "dummy", "binary"]:
                    transformer = self.encoders[column]
                    encoded_array = transformer.transform(df_transformed[[column]])
                    encoded_df = pd.DataFrame(
                        encoded_array,
                        columns=transformer.get_feature_names_out([column]),
                    )

                    df_transformed = pd.concat(
                        [df_transformed, encoded_df], axis=1
                    ).drop(columns=[column])

                elif transformation == "count":
                    df_transformed[column] = df_transformed[column].map(
                        self.encoders[column]
                    )

        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
