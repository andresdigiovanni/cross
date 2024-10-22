import streamlit as st


class ColumnCastingEdit:
    def show_component(self, params):
        cast_options = params["cast_options"]
        n_cols = 2

        cols = st.columns((1,) * n_cols)

        markdown = ["| Column | New type |", "| --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, dtype_to_cast) in enumerate(cast_options.items()):
            markdowns[i % n_cols].append(f"| {column} | {dtype_to_cast} |")

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
