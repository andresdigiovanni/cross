import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import next_button
from cross.preprocessing.dtypes import numerical_columns
from cross.preprocessing.missing_values import MissingValuesHandler


class MissingValuesPage:
    def show_page(self):
        st.title("Missing Values Handling")
        st.write(
            "Handle missing values in your DataFrame. "
            "Available options include doing nothing, dropping rows with missing values, "
            "filling missing values with the mean, median, mode, zero, interpolate, etc."
        )

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
            return

        df = st.session_state["data"]

        num_columns = numerical_columns(df)

        # Actions for each column
        actions = {
            "Do nothing": "none",
            "Drop": "drop",
            "Fill with mean": "fill_mean",
            "Fill with median": "fill_median",
            "Fill with mode": "fill_mode",
            "Fill with 0": "fill_0",
            "Interpolate": "interpolate",
        }

        handling_options = {}
        missing_values = df.isnull().sum()

        for column in num_columns:
            st.markdown("""---""")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(column)
                st.write(f"Missing values: {missing_values[column]}")
                handling_options[column] = st.selectbox(
                    f"Action for {column}", actions.keys(), key=column
                )

            with col2:
                fig, ax = plt.subplots(figsize=(4, 2))
                sns.histplot(df[column], kde=True, ax=ax)
                st.pyplot(fig)

        st.markdown("""---""")

        # Convert button
        if st.button("Apply Actions"):
            try:
                handling_options_mapped = {
                    col: actions[action] for col, action in handling_options.items()
                }
                missing_values_handler = MissingValuesHandler(handling_options_mapped)
                df = missing_values_handler.fit_transform(df)
                st.session_state["data"] = df

                config = st.session_state.get("config", {})
                config["missing_values"] = {
                    "handling_options": missing_values_handler.handling_options.copy(),
                    "statistics": missing_values_handler.statistics_.copy(),
                }
                st.session_state["config"] = config

                st.success("Missing values handled successfully!")

            except Exception as e:
                st.error("Error handling missing values: {}".format(e))

        # Next button
        next_button()
