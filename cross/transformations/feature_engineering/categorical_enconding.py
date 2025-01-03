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

        X = X.copy()

        for column, transformation in self.encodings_options.items():
            self._fit_encoder(X, column, transformation, y)

        return self

    def _fit_encoder(self, X, column, transformation, y):
        if transformation == "label":
            self._encoders[column] = LabelEncoder().fit(X[column])

        elif transformation == "ordinal":
            self._encoders[column] = OrdinalEncoder(
                categories=[self.ordinal_orders[column]]
            ).fit(X[[column]])

        elif transformation in ["onehot", "dummy"]:
            drop = "first" if transformation == "dummy" else None
            self._encoders[column] = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore", drop=drop
            ).fit(X[[column]])

        elif transformation == "binary":
            self._encoders[column] = BinaryEncoder().fit(X[[column]])

        elif transformation == "target" and y is not None:
            self._encoders[column] = TargetEncoder().fit(X[[column]], y)

        elif transformation == "count":
            self._encoders[column] = X[column].value_counts().to_dict()

    def _safe_transform(self, value, transformer, known_classes):
        return transformer.transform([value])[0] if value in known_classes else -1

    def transform(self, X, y=None):
        X = X.copy()

        for column, transformation in self.encodings_options.items():
            X[column] = X[column].fillna("Unknown")

            if column in self._encoders:
                transformer = self._encoders[column]
                X = self._transform_column(X, column, transformation, transformer)

        return X

    def _transform_column(self, X, column, transformation, transformer):
        if transformation == "label":
            known_classes = set(transformer.classes_)
            X[column] = X[column].apply(
                lambda val: self._safe_transform(val, transformer, known_classes)
            )

        elif transformation in ["ordinal", "target"]:
            X[column] = transformer.transform(X[[column]])

        elif transformation in ["onehot", "dummy", "binary"]:
            encoded_array = transformer.transform(X[[column]])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=transformer.get_feature_names_out([column]),
                index=X.index,
            )
            X = pd.concat([X, encoded_df], axis=1).drop(columns=[column])

        elif transformation == "count":
            X[column] = X[column].map(self._encoders[column]).fillna(0)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
