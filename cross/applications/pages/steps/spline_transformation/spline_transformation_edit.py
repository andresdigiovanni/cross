import streamlit as st

from .spline_transformation import SplineTransformationsBase


class SplineTransformationEdit(SplineTransformationsBase):
    def show_component(self, params):
        transformation_options = params["transformation_options"]
        n_cols = 2

        cols = st.columns((1,) * n_cols)

        markdown = ["| Column | Degree | N knots |", "| --- | --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, options) in enumerate(transformation_options.items()):
            markdowns[i % n_cols].append(
                "| {} | {} | {} |".format(column, options["degree"], options["n_knots"])
            )

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
