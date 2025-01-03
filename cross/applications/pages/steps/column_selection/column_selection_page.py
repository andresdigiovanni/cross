import streamlit as st

from cross.applications.components import is_data_loaded
from cross.transformations import ColumnSelection


class ColumnSelectionPage:
    def show_page(self):
        st.title("Column Selection")
        st.write("Select the columns you want to include in your analysis.")

        if not is_data_loaded():
            return

        df = st.session_state["data"]

        # Display data preview
        st.write(df.head())
        st.markdown("""---""")

        # Identify columns with a single unique value
        self._warn_single_value_columns(df)

        # Configure columns for selection
        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)
        columns = [col for col in df.columns if col != target_column]

        # Multi-select columns to include
        selected_columns = st.multiselect(
            "Select columns", options=columns, default=columns
        )

        # Apply column selection
        if st.button("Add step"):
            self._apply_column_selection(df, selected_columns)

    def _warn_single_value_columns(self, df):
        single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
        if single_value_columns:
            st.warning(
                f"The following columns have a single unique value and may not provide useful information: {', '.join(single_value_columns)}"
            )

    def _apply_column_selection(self, df, selected_columns):
        column_selector = ColumnSelection(selected_columns)
        transformed_df = column_selector.fit_transform(df)
        st.session_state["data"] = transformed_df

        # Save the transformation step
        steps = st.session_state.get("steps", [])
        steps.append(
            {"name": "ColumnSelection", "params": column_selector.get_params()}
        )
        st.session_state["steps"] = steps

        st.success("Columns selected successfully!")
