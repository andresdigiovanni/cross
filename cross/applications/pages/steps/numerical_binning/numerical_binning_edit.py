import streamlit as st

from .numerical_binning import NumericalBinningBase


class NumericalBinningEdit(NumericalBinningBase):
    def show_component(self, params):
        binning_options = params["binning_options"]
        num_bins = params["num_bins"]
        reverse_binnings = {v: k for k, v in self.binnings.items()}
        n_cols = 2

        cols = st.columns((1,) * n_cols)

        markdown = ["| Column | Strategy | Number of bins", "| --- | --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, strategy) in enumerate(binning_options.items()):
            if strategy == "none":
                continue

            markdowns[i % n_cols].append(
                "| {} | {} | {} |".format(
                    column, reverse_binnings[strategy], num_bins[column]
                )
            )

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
