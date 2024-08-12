import streamlit as st


class CyclicalFeaturesTransformationEdit:
    def show_component(self, params):
        columns_periods = params["columns_periods"]
        n_cols = 2

        cols = st.columns((1,) * n_cols)

        markdown = ["| Column | Period |", "| --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, transformation) in enumerate(columns_periods.items()):
            markdowns[i % n_cols].append("| {} | {} |".format(column, transformation))

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
