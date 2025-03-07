import streamlit as st
from streamlit_sortables import sort_items

from cross.applications.components import is_data_loaded
from cross.transformations import CategoricalEncoding
from cross.transformations.utils.dtypes import categorical_columns

from .categorical_encoding import CategoricalEncodingBase


class CategoricalEncodingPage(CategoricalEncodingBase):
    def show_page(self):
        st.title("Categorical Encoding")
        st.write("Select the encoding technique for each column.")

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        original_df = df.copy()
        cat_columns = categorical_columns(df)

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        # Initialize encoding options and ordinal orders
        encodings_options, ordinal_orders = self._initialize_encoding_options(
            cat_columns, original_df
        )

        # Apply button for encoding
        if st.button("Add step"):
            self._apply_encoding_step(
                encodings_options, ordinal_orders, df, target_column
            )

    def _initialize_encoding_options(self, cat_columns, original_df):
        encodings_options = {}
        ordinal_orders = {}

        for column in cat_columns:
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                self._display_column_info(
                    column, original_df, encodings_options, ordinal_orders
                )

            with col2:
                self._display_original_data(column, original_df)

            with col3:
                self._display_transformed_data(
                    column, original_df, encodings_options, ordinal_orders
                )

        return encodings_options, ordinal_orders

    def _display_column_info(
        self, column, original_df, encodings_options, ordinal_orders
    ):
        st.subheader(column)
        num_categories = original_df[column].nunique()
        st.write(f"Number of categories: {num_categories}")

        selected_encoding = st.selectbox(
            f"Select encoding for {column}",
            self.ENCODING_OPTIONS.keys(),
            key=column,
        )
        encodings_options[column] = selected_encoding

        # Handle ordinal encoding option
        if self.ENCODING_OPTIONS[selected_encoding] == "ordinal":
            categories = original_df[column].fillna("Unknown").unique().tolist()
            st.write("Order the categories")
            ordered_categories = sort_items(categories, key=f"{column}_order")
            ordinal_orders[column] = ordered_categories

    def _display_original_data(self, column, original_df):
        st.write("Original Data")
        st.dataframe(original_df[[column]].drop_duplicates().head())

    def _display_transformed_data(
        self, column, original_df, encodings_options, ordinal_orders
    ):
        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        encoding_type = self.ENCODING_OPTIONS[encodings_options[column]]
        if encoding_type != "none" and not (
            encoding_type in ["target", "loo"] and not target_column
        ):
            categorical_encoding = CategoricalEncoding(
                {column: encoding_type},
                ordinal_orders=ordinal_orders,
            )
            transformed_df = categorical_encoding.fit_transform(
                original_df.loc[:, ~original_df.columns.isin([target_column])],
                original_df[target_column] if target_column else None,
            )

            # Determine if new columns are created
            new_columns = (
                list(set(transformed_df.columns) - set(original_df.columns))
                if encoding_type
                in ["onehot", "dummy", "binary", "target", "loo", "rankhot"]
                else [column]
            )

            st.write("Transformed Data")
            st.dataframe(transformed_df[new_columns].drop_duplicates().head())
        else:
            st.write("No transformation applied")

    def _apply_encoding_step(
        self, encodings_options, ordinal_orders, df, target_column
    ):
        try:
            # Map valid encodings
            encodings_mapped = {
                col: self.ENCODING_OPTIONS[encoding]
                for col, encoding in encodings_options.items()
                if self.ENCODING_OPTIONS[encoding] != "none"
            }

            # Apply categorical encoding
            categorical_encoding = CategoricalEncoding(
                encodings_mapped, ordinal_orders=ordinal_orders
            )
            transformed_df = categorical_encoding.fit_transform(df)

            # Update session state
            st.session_state["data"] = transformed_df
            params = categorical_encoding.get_params()

            # Append step to session state
            steps = st.session_state.get("steps", [])
            steps.append({"name": "CategoricalEncoding", "params": params})
            st.session_state["steps"] = steps

            st.success("Encoding applied successfully!")

        except Exception as e:
            st.error(f"Error applying encoding: {e}")
