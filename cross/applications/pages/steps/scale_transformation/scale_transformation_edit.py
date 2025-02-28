import streamlit as st

from .scale_transformation import ScaleTransformationsBase


class ScaleTransformationsEdit(ScaleTransformationsBase):
    def show_component(self, params):
        transformation_options = params["transformation_options"]
        quantile_range = params["quantile_range"]

        reverse_transformations = {v: k for k, v in self.TRANSFORMATIONS.items()}
        n_cols = 2

        cols = st.columns((1,) * n_cols)

        markdown = [
            "| Column | Transformation | Quantile range |",
            "| --- | --- | --- |",
        ]
        markdowns = [markdown.copy() for _ in range(n_cols)]

        for i, (column, transformation) in enumerate(transformation_options.items()):
            if transformation == "none":
                continue

            markdowns[i % n_cols].append(
                "| {} | {} | {} |".format(
                    column,
                    reverse_transformations[transformation],
                    quantile_range.get(column, ""),
                )
            )

        for col, markdown in zip(cols, markdowns):
            with col:
                if len(markdown) > 2:
                    st.markdown("\n".join(markdown))
