import streamlit as st

from .numerical_binning import NumericalBinningBase


class NumericalBinningEdit(NumericalBinningBase):
    def show_component(self, params):
        binning_options = params["binning_options"]
        reverse_binnings = {v: k for k, v in self.BINNINGS.items()}
        n_cols = 2

        cols = st.columns((1,) * n_cols)

        markdown = ["| Column | Strategy | Number of bins", "| --- | --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, strategy, n_bins) in enumerate(binning_options):
            if strategy == "none":
                continue

            markdowns[i % n_cols].append(
                "| {} | {} | {} |".format(column, reverse_binnings[strategy], n_bins)
            )

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
