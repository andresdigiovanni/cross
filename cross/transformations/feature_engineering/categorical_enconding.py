import pandas as pd
from category_encoders import BinaryEncoder, TargetEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


class CategoricalEncoding:
    def __init__(
        self,
        encodings_options=None,
        ordinal_orders=None,
        encoders=None,
    ):
        self.encodings_options = encodings_options or {}
        self.ordinal_orders = ordinal_orders or {}
        self.encoders = encoders or {}

    def get_params(self):
        return {
            "encodings_options": self.encodings_options,
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

            elif transformation == "target":
                if y is not None:
                    transformer = TargetEncoder()
                    transformer.fit(x_filled[[column]], y)
                    self.encoders[column] = transformer

            elif transformation == "count":
                self.encoders[column] = x_filled[column].value_counts().to_dict()

    def _safe_transform(self, value, transformer, known_classes):
        if value in known_classes:
            return transformer.transform([value])[0]

        else:
            return -1

    def transform(self, x, y=None):
        x_transformed = x.copy()
        y_transformed = y.copy() if y is not None else None

        x_transformed = x_transformed.fillna("Unknown")

        for column, transformation in self.encodings_options.items():
            if column in self.encoders:
                transformer = self.encoders[column]

                if transformation == "label":
                    known_classes = set(transformer.classes_)
                    x_transformed[column] = x_transformed[column].apply(
                        lambda val: self._safe_transform(
                            val, transformer, known_classes
                        )
                    )

                elif transformation in ["ordinal", "target"]:
                    x_transformed[column] = transformer.transform(
                        x_transformed[[column]]
                    )

                elif transformation in ["onehot", "dummy", "binary"]:
                    encoded_array = transformer.transform(x_transformed[[column]])
                    encoded_df = pd.DataFrame(
                        encoded_array,
                        columns=transformer.get_feature_names_out([column]),
                    )

                    encoded_df.index = x_transformed.index
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
