import streamlit as st

from cross.applications.components import next_button
from cross.preprocessing.column_selection import column_selection


def show_page():
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
        df = column_selection(df, selected_columns)
        st.session_state["data"] = df

        config = st.session_state.get("config", {})
        config["selected_columns"] = selected_columns
        st.session_state["config"] = config

        st.success("Columns selected successfully!")

    # Next button
    next_button()
