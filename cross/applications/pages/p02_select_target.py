import streamlit as st

from cross.applications.components import next_button


def update_target_column():
    target_column = st.session_state["target_column_selectbox"]
    target_column = "" if target_column == "None" else target_column

    config = st.session_state.get("config", {})
    config["target_column"] = target_column
    st.session_state["config"] = config


def show_page():
    st.title("Select Target Column")
    st.write(
        "Choose the target column for your analysis. This can be useful for tasks such as classification, "
        "regression, etc. You also have the option to proceed without selecting a target column."
    )

    if "data" not in st.session_state:
        st.warning("No data loaded. Please load a DataFrame.")
        return

    df = st.session_state["data"]
    columns = df.columns.tolist()
    columns.insert(0, "None")  # Add the option to select no target column

    st.selectbox(
        "Select the target column:",
        options=columns,
        index=0,
        key="target_column_selectbox",
        on_change=update_target_column,
    )

    # Next button
    next_button()
