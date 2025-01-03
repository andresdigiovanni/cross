import streamlit as st

from cross.applications.components import is_data_loaded
from cross.transformations import NumericalBinning
from cross.transformations.utils.dtypes import numerical_columns

from .numerical_binning import NumericalBinningBase


class NumericalBinningPage(NumericalBinningBase):
    def show_page(self):
        st.title("Numerical Binning")
        st.write("Select the binning technique for each column.")

        if not is_data_loaded():
            return

        df, num_columns, original_df, target_column = self._initialize_data()

        binning_options = {}
        num_bins = {}

        self._display_binning_options(
            df, num_columns, original_df, binning_options, num_bins
        )

        st.markdown("""---""")

        self._apply_binning(df, binning_options, num_bins)

    def _initialize_data(self):
        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = numerical_columns(df)
        num_columns = [x for x in num_columns if x != target_column]

        return df, num_columns, original_df, target_column

    def _display_binning_options(
        self, df, num_columns, original_df, binning_options, num_bins
    ):
        for column in num_columns:
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(column)
                selected_binning = st.selectbox(
                    f"Select binning for {column}",
                    self.BINNINGS.keys(),
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
                if self.BINNINGS[binning_options[column]] != "none":
                    transformed_df = self._apply_single_binning(
                        original_df, column, binning_options[column], num_bins[column]
                    )
                    new_column = f"{column}__{self.BINNINGS[binning_options[column]]}_{num_bins[column]}"
                    st.write("Binned Data")
                    st.dataframe(transformed_df[[new_column]].head())
                else:
                    st.write("No binning applied")

    def _apply_single_binning(self, df, column, binning, bins):
        numerical_binning = NumericalBinning([(column, self.BINNINGS[binning], bins)])
        return numerical_binning.fit_transform(df)

    def _apply_binning(self, df, binning_options, num_bins):
        if st.button("Add step"):
            try:
                binnings_mapped = [
                    (column, self.BINNINGS[binning], num_bins[column])
                    for column, binning in binning_options.items()
                    if self.BINNINGS[binning] != "none"
                ]
                numerical_binning = NumericalBinning(binnings_mapped)

                transformed_df = numerical_binning.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = numerical_binning.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "NumericalBinning", "params": params})
                st.session_state["steps"] = steps

                st.success("Binning applied successfully!")

            except Exception as e:
                st.error(f"Error applying binning: {e}")
