import streamlit as st

from cross.applications.components import is_data_loaded
from cross.transformations import CorrelatedSubstringEncoder
from cross.transformations.utils.dtypes import categorical_columns


class CorrelatedSubstringPage:
    def show_page(self):
        st.title("Correlated Substring Encoding")
        st.write(
            "Configure the correlated substrings for each column and apply the transformations."
        )

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        original_df = df.copy()
        cat_columns = categorical_columns(df)

        # Initialize operation options if not present
        if "correlated_substrings" not in st.session_state:
            st.session_state.correlated_substrings = {cat_columns[0]: ""}

        self._display_operations(cat_columns, original_df)
        st.button("Add another operation", on_click=self._add_operation)
        st.markdown("""---""")

        self._apply_operations(df)

    def _add_operation(self):
        used_columns = set(st.session_state.correlated_substrings.keys())
        cat_columns = [
            col
            for col in categorical_columns(st.session_state["data"])
            if col not in used_columns
        ]

        if len(cat_columns):
            st.session_state.correlated_substrings[cat_columns[0]] = ""

    def _display_operations(self, cat_columns, original_df):
        for i, (column, substrings) in enumerate(
            st.session_state.correlated_substrings.items()
        ):
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            column, substrings = self._select_operation(
                i, column, substrings, cat_columns, col1
            )
            st.session_state.correlated_substrings[column] = substrings

            self._preview_original_data(original_df, column, col2)
            self._preview_transformed_data(original_df, column, substrings, col3)

    def _select_operation(self, i, column, substrings, cat_columns, col):
        with col:
            column = st.selectbox(
                f"Select column for operation {i + 1}",
                cat_columns,
                index=cat_columns.index(column) if column in cat_columns else 0,
                key=f"column_{i}",
            )
            substrings = st.text_area(
                f"Substrings for {column} (comma-separated)",
                value=substrings,
                key=f"{column}_substrings",
            )

            return column, substrings

    def _preview_original_data(self, original_df, column, col):
        with col:
            st.write("Original Data")
            st.dataframe(original_df[[column]].head())

    def _preview_transformed_data(self, original_df, column, substrings, col):
        with col:
            if substrings:
                substrings_list = self._split_substrings(substrings)
                encoder = CorrelatedSubstringEncoder(
                    substrings={column: substrings_list}
                )
                transformed_df = encoder.fit_transform(original_df[[column]])

                new_column = list(
                    set(transformed_df.columns) - set(original_df.columns)
                )
                st.write("Transformed Data")
                st.dataframe(transformed_df[new_column].head())
            else:
                st.write("No transformation applied")

    def _apply_operations(self, df):
        if st.button("Add step"):
            try:
                substrings_config = st.session_state.correlated_substrings
                substrings_config = {
                    col: self._split_substrings(substrings)
                    for col, substrings in substrings_config.items()
                }
                substrings_config = {k: v for k, v in substrings_config.items() if v}

                # Apply correlated substring encoding
                encoder = CorrelatedSubstringEncoder(substrings=substrings_config)
                transformed_df = encoder.fit_transform(df)
                params = encoder.get_params()

                # Update session state
                st.session_state["data"] = transformed_df

                # Append step to session state
                steps = st.session_state.get("steps", [])
                steps.append({"name": "CorrelatedSubstringEncoder", "params": params})
                st.session_state["steps"] = steps

                # Reset operation options
                st.session_state.correlated_substrings = {}

                st.success("Encoding applied successfully!")

            except Exception as e:
                st.error(f"Error applying operations: {e}")

    def _split_substrings(self, substrings):
        return [s.strip() for s in substrings.split(",") if s.strip()]
