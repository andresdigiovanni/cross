import streamlit as st

from .mathematical_operations import MathematicalOperationsBase


class MathematicalOperationsEdit(MathematicalOperationsBase):
    def show_component(self, params):
        operations_options = params["operations_options"]
        reverse_operations = {v: k for k, v in self.encodings.items()}
        n_cols = 2

        cols = st.columns((1,) * n_cols)

        markdown = ["| Column 1 | Column 2 | Action", "| --- | --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (col1, col2, operation) in enumerate(operations_options):
            if operation == "none":
                continue

            markdowns[i % n_cols].append(
                "| {} | {} | {} |".format(col1, col2, reverse_operations[operation])
            )

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
