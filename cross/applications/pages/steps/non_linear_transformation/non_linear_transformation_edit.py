import streamlit as st

from .non_linear_transformation import NonLinearTransformationBase


class NonLinearTransformationEdit(NonLinearTransformationBase):
    def show_component(self, params):
        transformation_options = params["transformation_options"]
        reverse_transformations = {v: k for k, v in self.transformations.items()}
        n_cols = 2

        cols = st.columns((1,) * n_cols)

        markdown = ["| Column | Transformation |", "| --- | --- |"]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, transformation) in enumerate(transformation_options.items()):
            if transformation == "none":
                continue

            markdowns[i % n_cols].append(
                "| {} | {} |".format(column, reverse_transformations[transformation])
            )

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
