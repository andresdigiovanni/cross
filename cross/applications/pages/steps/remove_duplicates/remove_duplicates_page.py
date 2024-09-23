import streamlit as st

from cross.transformations.clean_data import RemoveDuplicatesHandler

from .remove_duplicates import RemoveDuplicatesBase


class RemoveDuplicatesPage(RemoveDuplicatesBase):
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

        # Select columns to consider for identifying duplicates
        st.subheader("Select Columns to Consider for Identifying Duplicates")

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        columns = [x for x in df.columns if x != target_column]
        subset = st.multiselect("Columns", options=columns, default=columns)

        # Display initial number of duplicates
        initial_duplicates = df.duplicated().sum()
        st.write(f"Initial number of duplicate rows: {initial_duplicates}")

        # Select how to handle duplicates
        st.subheader("Select How to Handle Duplicates")
        keep = st.selectbox("Action", list(self.keep_options.keys()))
        keep = self.keep_options[keep]

        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                remove_duplicates_handler = RemoveDuplicatesHandler(
                    subset=subset, keep=keep
                )
                transformed_df = remove_duplicates_handler.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = remove_duplicates_handler.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "RemoveDuplicatesHandler", "params": params})
                st.session_state["steps"] = steps

                st.success("Duplicates removed successfully!")

            except Exception as e:
                st.error("Error removing duplicates: {}".format(e))
