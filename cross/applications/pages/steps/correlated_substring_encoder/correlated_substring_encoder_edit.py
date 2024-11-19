import streamlit as st


class CorrelatedSubstringEdit:
    def show_component(self, params):
        substrings_config = params.get("substrings", {})

        n_cols = 2
        cols = st.columns((1,) * n_cols)

        markdown = ["| Column | Substrings |", "| --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, substrings) in enumerate(substrings_config.items()):
            markdowns[i % n_cols].append(
                "| {} | {} |".format(column, ", ".join(substrings))
            )

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
