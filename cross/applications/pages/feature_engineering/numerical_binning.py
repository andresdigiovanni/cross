from copy import deepcopy

import streamlit as st

from cross.applications.components import next_button
from cross.feature_engineering.numerical_binning import NumericalBinning
from cross.preprocessing.dtypes import numerical_columns


class NumericalBinningPage:
    def show_page(self):
        st.title("Numerical Binning")
        st.write("Select the binning technique for each column.")

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
            return

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = numerical_columns(df)

        # Actions for each column
        binnings = {
            "Do nothing": "none",
            "Uniform Binning": "uniform",
            "Quantile Binning": "quantile",
            "KMeans Binning": "kmeans",
        }

        binning_options = {}
        num_bins = {}

        for column in num_columns:
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(column)
                selected_binning = st.selectbox(
                    f"Select binning for {column}",
                    binnings.keys(),
                    key=f"{column}_binning",
                )
                binning_options[column] = selected_binning

                if selected_binning != "none":
                    bins = st.slider(
                        f"Select number of bins for {column}",
                        min_value=2,
                        max_value=20,
                        value=5,
                        key=f"{column}_bins",
                    )
                    num_bins[column] = bins

            with col2:
                st.write("Original Data")
                st.dataframe(original_df[[column]].head())

            with col3:
                if binnings[binning_options[column]] != "none":
                    numerical_binning = NumericalBinning(
                        {column: binnings[binning_options[column]]}, num_bins
                    )
                    transformed_df = numerical_binning.fit_transform(original_df)

                    st.write("Binned Data")
                    st.dataframe(transformed_df[[column]].head())
                else:
                    st.write("No binning applied")

        st.markdown("""---""")

        # Apply button
        if st.button("Apply binning"):
            try:
                binnings_mapped = {
                    col: binnings[binning] for col, binning in binning_options.items()
                }

                numerical_binning = NumericalBinning(binnings_mapped, num_bins)
                transformed_df = numerical_binning.fit_transform(original_df)
                st.session_state["data"] = transformed_df

                config = st.session_state.get("config", {})
                config["numerical_binning"] = {
                    "binning_options": deepcopy(numerical_binning.binning_options),
                    "num_bins": deepcopy(numerical_binning.num_bins),
                }
                st.session_state["config"] = config

                st.success("Binning applied successfully!")

            except Exception as e:
                st.error("Error applying binning: {}".format(e))

        # Next button
        next_button()
