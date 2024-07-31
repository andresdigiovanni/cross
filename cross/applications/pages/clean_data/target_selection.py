import streamlit as st

from cross.applications.components.check_is_data_loaded import is_data_loaded


class TargetSelectionPage:
    def show_page(self, name):
        st.title("Select Target Column")
        st.write(
            "Choose the target column for your analysis. This can be useful for tasks such as classification, "
            "regression, etc. You also have the option to proceed without selecting a target column."
        )

        if not is_data_loaded():
            return

        df = st.session_state["data"]

        # Display data
        st.write(df.head())
        st.markdown("""---""")

        columns = df.columns.tolist()
        columns.insert(0, "None")  # Add the option to select no target column

        st.selectbox(
            "Select the target column:",
            options=columns,
            index=0,
            key="target_column_selectbox",
            on_change=self._update_target_column,
        )

    def _update_target_column(self):
        target_column = st.session_state["target_column_selectbox"]
        target_column = "" if target_column == "None" else target_column

        config = st.session_state.get("config", {})
        config["target_column"] = target_column
        st.session_state["config"] = config
