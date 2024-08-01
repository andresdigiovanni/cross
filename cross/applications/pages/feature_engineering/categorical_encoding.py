import streamlit as st
from streamlit_sortables import sort_items

from cross.applications.components import is_data_loaded
from cross.core.feature_engineering.categorical_enconding import CategoricalEncoding
from cross.core.utils.dtypes import categorical_columns


class CategoricalEncodingPage:
    def show_page(self, name):
        st.title("Categorical Encoding")
        st.write("Select the encoding technique for each column.")

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        original_df = df.copy()

        cat_columns = categorical_columns(df)

        # Actions for each column
        encodings = {
            "Do nothing": "none",
            "Label Encoder": "label",
            "Ordinal Encoder": "ordinal",
            "One Hot Encoder": "onehot",
            "Dummy Encoder": "dummy",
            "Binary Encoder": "binary",
            "Count Encoder": "count",
            "Target Encoder": "target",
        }

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        encodings_options = {}
        ordinal_orders = {}

        for column in cat_columns:
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(column)

                num_categories = original_df[column].nunique()
                st.write(f"Number of categories: {num_categories}")

                selected_encoding = st.selectbox(
                    f"Select encoding for {column}",
                    encodings.keys(),
                    key=column,
                )
                encodings_options[column] = selected_encoding

                if encodings[selected_encoding] == "ordinal":
                    categories = original_df[column].fillna("Unknown").unique().tolist()
                    st.write("Order the categories")

                    ordered_categories = sort_items(categories, key=f"{column}_order")
                    ordinal_orders[column] = ordered_categories

            with col2:
                st.write("Original Data")
                st.dataframe(original_df[[column]].drop_duplicates().head())

            with col3:
                if (encodings[encodings_options[column]] != "none") and not (
                    encodings[encodings_options[column]] == "target"
                    and (target_column is None or target_column == "")
                ):
                    categorical_encoding = CategoricalEncoding(
                        {column: encodings[encodings_options[column]]},
                        target_column=target_column,
                        ordinal_orders=ordinal_orders,
                    )
                    transformed_df = categorical_encoding.fit_transform(original_df)

                    # If multiple columns are created
                    if encodings[encodings_options[column]] in [
                        "onehot",
                        "dummy",
                        "binary",
                    ]:
                        new_columns = list(
                            set(transformed_df.columns) - set(original_df.columns)
                        )
                    else:
                        new_columns = [column]

                    st.write("Transformed Data")
                    st.dataframe(transformed_df[new_columns].drop_duplicates().head())
                else:
                    st.write("No transformation applied")

        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                encodings_mapped = {
                    col: encodings[transformation]
                    for col, transformation in encodings_options.items()
                }

                categorical_encoding = CategoricalEncoding(
                    encodings_mapped,
                    target_column=target_column,
                    ordinal_orders=ordinal_orders,
                )
                transformed_df = categorical_encoding.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = categorical_encoding.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": name, "params": params})
                st.session_state["steps"] = steps

                st.success("Encoding applied successfully!")

            except Exception as e:
                st.error("Error applying encoding: {}".format(e))
