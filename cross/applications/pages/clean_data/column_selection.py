import streamlit as st

from cross.applications.components import next_button
from cross.core.clean_data.column_selection import ColumnSelection


class ColumnSelectionPage:
    def show_page(self):
        st.title("Column Selection")
        st.write("Select the columns you want to include in your analysis.")

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
            return

        df = st.session_state["data"]

        # Display data
        st.write(df.head())
        st.markdown("""---""")

        # Selección de columnas
        selected_columns = st.multiselect(
            "Select columns", options=df.columns.tolist(), default=df.columns.tolist()
        )

        if st.button("Apply Selection"):
            column_selector = ColumnSelection(selected_columns)
            transformed_df = column_selector.fit_transform(df)
            st.session_state["data"] = transformed_df

            config = st.session_state.get("config", {})
            config["column_selector"] = column_selector.get_params()
            st.session_state["config"] = config

            st.success("Columns selected successfully!")

        # Next button
        next_button()
