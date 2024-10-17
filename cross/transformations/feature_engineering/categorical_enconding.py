import pandas as pd
from category_encoders import BinaryEncoder, TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


class CategoricalEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, encodings_options=None, ordinal_orders=None):
        self.encodings_options = encodings_options or {}
        self.ordinal_orders = ordinal_orders

        self._encoders = {}

    def get_params(self, deep=True):
        return {
            "encodings_options": self.encodings_options,
            "ordinal_orders": self.ordinal_orders,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._encoders = {}

        X_filled = X.copy()

        for column, transformation in self.encodings_options.items():
            X_filled[column] = X_filled[column].fillna("Unknown")

            if transformation == "label":
                transformer = LabelEncoder()
                transformer.fit(X_filled[column])
                self._encoders[column] = transformer

            elif transformation == "ordinal":
                transformer = OrdinalEncoder(categories=[self.ordinal_orders[column]])
                transformer.fit(X_filled[[column]])
                self._encoders[column] = transformer

            elif transformation == "onehot":
                transformer = OneHotEncoder(sparse_output=False)
                transformer.fit(X_filled[[column]])
                self._encoders[column] = transformer

            elif transformation == "dummy":
                transformer = OneHotEncoder(sparse_output=False, drop="first")
                transformer.fit(X_filled[[column]])
                self._encoders[column] = transformer

            elif transformation == "binary":
                transformer = BinaryEncoder()
                transformer.fit(X_filled[[column]])
                self._encoders[column] = transformer

            elif transformation == "target" and y is not None:
                transformer = TargetEncoder()
                transformer.fit(X_filled[[column]], y)
                self._encoders[column] = transformer

            elif transformation == "count":
                self._encoders[column] = X_filled[column].value_counts().to_dict()

        return self

    def _safe_transform(self, value, transformer, known_classes):
        return transformer.transform([value])[0] if value in known_classes else -1

    def transform(self, X, y=None):
        X_transformed = X.copy()

        for column, transformation in self.encodings_options.items():
            X_transformed[column] = X_transformed[column].fillna("Unknown")

            if column in self._encoders:
                transformer = self._encoders[column]

                if transformation == "label":
                    known_classes = set(transformer.classes_)
                    X_transformed[column] = X_transformed[column].apply(
                        lambda val: self._safe_transform(
                            val, transformer, known_classes
                        )
                    )

                elif transformation in ["ordinal", "target"]:
                    X_transformed[column] = transformer.transform(
                        X_transformed[[column]]
                    )

                elif transformation in ["onehot", "dummy", "binary"]:
                    encoded_array = transformer.transform(X_transformed[[column]])
                    encoded_df = pd.DataFrame(
                        encoded_array,
                        columns=transformer.get_feature_names_out([column]),
                    )
                    encoded_df.index = X_transformed.index
                    X_transformed = pd.concat([X_transformed, encoded_df], axis=1).drop(
                        columns=[column]
                    )

                elif transformation == "count":
                    X_transformed[column] = (
                        X_transformed[column].map(self._encoders[column]).fillna(0)
                    )

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
