import pandas as pd
from category_encoders import BinaryEncoder, TargetEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


class CategoricalEncoding:
    def __init__(
        self,
        encodings_options=None,
        target_column=None,
        ordinal_orders=None,
        encoders=None,
    ):
        self.encodings_options = encodings_options or {}
        self.target_column = target_column or ""
        self.ordinal_orders = ordinal_orders or {}
        self.encoders = encoders or {}

    def get_params(self):
        return {
            "encodings_options": self.encodings_options,
            "target_column": self.target_column,
            "ordinal_orders": self.ordinal_orders,
            "encoders": self.encoders,
        }

    def fit(self, x, y=None):
        self.encoders = {}

        x_filled = x.copy()
        x_filled = x_filled.fillna("Unknown")

        for column, transformation in self.encodings_options.items():
            if transformation == "label":
                transformer = LabelEncoder()
                transformer.fit(x_filled[column])
                self.encoders[column] = transformer

            elif transformation == "ordinal":
                transformer = OrdinalEncoder(categories=[self.ordinal_orders[column]])
                transformer.fit(x_filled[[column]])
                self.encoders[column] = transformer

            elif transformation == "onehot":
                transformer = OneHotEncoder(sparse_output=False)
                transformer.fit(x_filled[[column]])
                self.encoders[column] = transformer

            elif transformation == "dummy":
                transformer = OneHotEncoder(sparse_output=False, drop="first")
                transformer.fit(x_filled[[column]])
                self.encoders[column] = transformer

            elif transformation == "binary":
                transformer = BinaryEncoder()
                transformer.fit(x_filled[[column]])
                self.encoders[column] = transformer

            elif (
                transformation == "target"
                and self.target_column is not None
                and self.target_column != ""
            ):
                transformer = TargetEncoder()
                transformer.fit(x_filled[[column]], x_filled[self.target_column])
                self.encoders[column] = transformer

            elif transformation == "count":
                self.encoders[column] = x_filled[column].value_counts().to_dict()

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        x_transformed = x_transformed.fillna("Unknown")

        for column, transformation in self.encodings_options.items():
            if column in self.encoders:
                if transformation in ["label", "ordinal", "target"]:
                    transformer = self.encoders[column]
                    x_transformed[column] = transformer.transform(
                        x_transformed[[column]]
                    )

                elif transformation in ["onehot", "dummy", "binary"]:
                    transformer = self.encoders[column]
                    encoded_array = transformer.transform(x_transformed[[column]])
                    encoded_df = pd.DataFrame(
                        encoded_array,
                        columns=transformer.get_feature_names_out([column]),
                    )

                    x_transformed = pd.concat([x_transformed, encoded_df], axis=1).drop(
                        columns=[column]
                    )

                elif transformation == "count":
                    x_transformed[column] = x_transformed[column].map(
                        self.encoders[column]
                    )

        if y_transformed is not None:
            return x_transformed, y_transformed
        else:
            return x_transformed

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
