import streamlit as st

from .categorical_encoding import CategoricalEncodingBase


class CategoricalEncodingEdit(CategoricalEncodingBase):
    def show_component(self, params):
        encodings_options = params["encodings_options"]
        reverse_encodings = {v: k for k, v in self.ENCODING_OPTIONS.items()}
        n_cols = 2
        cols = st.columns((1,) * n_cols)

        markdown = ["| Column | Transformation |", "| --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, transformation) in enumerate(encodings_options.items()):
            if transformation == "none":
                continue

            markdowns[i % n_cols].append(
                f"| {column} | {reverse_encodings[transformation]} |"
            )

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
