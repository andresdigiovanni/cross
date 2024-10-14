import streamlit as st

from .missing_values import MissingValuesBase


class MissingValuesEdit(MissingValuesBase):
    def show_component(self, params):
        handling_options = params["handling_options"]
        n_neighbors = params["n_neighbors"]
        reverse_actions = {v: k for k, v in self.actions.items()}
        n_cols = 2

        cols = st.columns((1,) * n_cols)

        markdown = ["| Column | Action | N neighbors", "| --- | --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, action) in enumerate(handling_options.items()):
            if action in [
                "fill_mean",
                "fill_median",
                "fill_mode",
                "fill_0",
                "interpolate",
                "most_frequent",
            ]:
                markdowns[i % n_cols].append(
                    "| {} | {} | |".format(column, reverse_actions[action])
                )

            elif action == "fill_knn":
                markdowns[i % n_cols].append(
                    "| {} | {} | {} |".format(
                        column, reverse_actions[action], n_neighbors[column]
                    )
                )

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
