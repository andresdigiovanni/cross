from copy import deepcopy

import streamlit as st

from cross.applications.components import next_button
from cross.feature_engineering.categorical_enconding import CategoricalEncoding
from cross.preprocessing.dtypes import categorical_columns


class CategoricalEncodingPage:
    def show_page(self):
        st.title("Categorical Encoding")
        st.write("Select the encoding technique for each column.")

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
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

            with col2:
                st.write("Original Data")
                st.dataframe(original_df[[column]].drop_duplicates().head())

            with col3:
                if (encodings_options[column] != "Do nothing") and not (
                    encodings_options[column] == "Target Encoder"
                    and (target_column is None or target_column == "")
                ):
                    categorical_encoding = CategoricalEncoding(
                        {column: encodings[encodings_options[column]]}, target_column
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
        if st.button("Apply encoding"):
            try:
                encodings_mapped = {
                    col: encodings[transformation]
                    for col, transformation in encodings_options.items()
                }

                categorical_encoding = CategoricalEncoding(
                    encodings_mapped, target_column
                )
                transformed_df = categorical_encoding.fit_transform(original_df)
                st.session_state["data"] = transformed_df

                config = st.session_state.get("config", {})
                config["categorical_encoding"] = {
                    "encodings_options": categorical_encoding.encodings_options.copy(),
                    "encoders": deepcopy(categorical_encoding.encoders),
                }
                st.session_state["config"] = config

                st.success("Encoding applied successfully!")

            except Exception as e:
                st.error("Error applying encoding: {}".format(e))

        # Next button
        next_button()
