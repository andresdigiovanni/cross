import streamlit as st

from cross.applications.components import is_data_loaded
from cross.core.clean_data import ColumnSelection


class ColumnSelectionPage:
    def show_page(self):
        st.title("Column Selection")
        st.write("Select the columns you want to include in your analysis.")

        if not is_data_loaded():
            return

        df = st.session_state["data"]

        # Display data
        st.write(df.head())
        st.markdown("""---""")

        # Check for columns with a single unique value
        single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
        if single_value_columns:
            st.warning(
                f"The following columns have a single unique value and may not provide useful information: {', '.join(single_value_columns)}"
            )

        # Columns selection
        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        columns = [x for x in df.columns if x != target_column]
        selected_columns = st.multiselect(
            "Select columns", options=columns, default=columns
        )

        if st.button("Add step"):
            column_selector = ColumnSelection(selected_columns)
            transformed_df = column_selector.fit_transform(df)
            st.session_state["data"] = transformed_df

            params = column_selector.get_params()
            steps = st.session_state.get("steps", [])
            steps.append({"name": "ColumnSelection", "params": params})
            st.session_state["steps"] = steps

            st.success("Columns selected successfully!")
