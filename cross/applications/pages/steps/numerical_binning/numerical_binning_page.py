import streamlit as st

from cross.applications.components import is_data_loaded
from cross.core.feature_engineering import NumericalBinning
from cross.core.utils.dtypes import numerical_columns

from .numerical_binning import NumericalBinningBase


class NumericalBinningPage(NumericalBinningBase):
    def show_page(self):
        st.title("Numerical Binning")
        st.write("Select the binning technique for each column.")

        if not is_data_loaded():
            return

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = numerical_columns(df)
        num_columns = [x for x in num_columns if x != target_column]

        binning_options = {}
        num_bins = {}

        for column in num_columns:
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(column)
                selected_binning = st.selectbox(
                    f"Select binning for {column}",
                    self.binnings.keys(),
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
                if self.binnings[binning_options[column]] != "none":
                    numerical_binning = NumericalBinning(
                        {column: self.binnings[binning_options[column]]}, num_bins
                    )
                    transformed_df = numerical_binning.fit_transform(original_df)

                    new_column = "{}__{}_{}".format(
                        column, self.binnings[binning_options[column]], num_bins[column]
                    )
                    st.write("Binned Data")
                    st.dataframe(transformed_df[[new_column]].head())
                else:
                    st.write("No binning applied")

        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                binnings_mapped = {
                    col: self.binnings[binning]
                    for col, binning in binning_options.items()
                }

                numerical_binning = NumericalBinning(binnings_mapped, num_bins)
                transformed_df = numerical_binning.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = numerical_binning.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "NumericalBinning", "params": params})
                st.session_state["steps"] = steps

                st.success("Binning applied successfully!")

            except Exception as e:
                st.error("Error applying binning: {}".format(e))
