import streamlit as st

from cross.applications.components import next_button
from cross.core.clean_data.remove_duplicates_handler import RemoveDuplicatesHandler


class RemoveDuplicatesPage:
    def show_page(self):
        st.title("Remove Duplicates")
        st.write(
            "Handle duplicate rows in your DataFrame. "
            "You can choose to keep the first occurrence, the last occurrence, or remove all duplicates."
        )

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
            return

        df = st.session_state["data"]

        # Display initial number of duplicates
        initial_duplicates = df.duplicated().sum()
        st.write(f"Initial number of duplicate rows: {initial_duplicates}")

        # Select columns to consider for identifying duplicates
        st.subheader("Select Columns to Consider for Identifying Duplicates")
        subset = st.multiselect(
            "Columns", options=df.columns.tolist(), default=df.columns.tolist()
        )

        # Select how to handle duplicates
        st.subheader("Select How to Handle Duplicates")
        keep_options = {
            "Do nothing": "none",
            "Keep first occurrence": "first",
            "Keep last occurrence": "last",
            "Remove all duplicates": False,
        }
        keep = st.selectbox("Action", list(keep_options.keys()))
        keep = keep_options[keep]

        st.markdown("""---""")

        # Apply button
        if st.button("Apply"):
            try:
                remove_duplicates_handler = RemoveDuplicatesHandler(
                    subset=subset, keep=keep
                )
                df = remove_duplicates_handler.fit_transform(df)
                st.session_state["data"] = df

                config = st.session_state.get("config", {})
                config["remove_duplicates"] = {"subset": subset, "keep": keep}
                st.session_state["config"] = config

                st.success("Duplicates removed successfully!")

            except Exception as e:
                st.error("Error removing duplicates: {}".format(e))

        # Next button
        next_button()
