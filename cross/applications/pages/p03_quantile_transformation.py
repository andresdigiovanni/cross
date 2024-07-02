from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import next_button
from cross.quantile_transformation import QuantileTransformation


def show_page():
    st.title("Quantile Transformations")
    st.write(
        "Select and apply quantile transformations (Uniform or Normal) to each column of your dataset."
    )

    if "data" not in st.session_state:
        st.warning("No data loaded. Please load a DataFrame.")
        return

    df = st.session_state["data"]
    original_df = df.copy()

    # Actions for each column
    transformations = {
        "Do nothing": "none",
        "Quantile Transformation (Uniform)": "uniform",
        "Quantile Transformation (Normal)": "normal",
    }

    transformation_options = {}

    for column in df.columns:
        st.markdown("""---""")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(column)
            selected_transformation = st.selectbox(
                f"Select transformation for {column}",
                transformations.keys(),
                key=column,
            )
            transformation_options[column] = selected_transformation

        with col2:
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.histplot(original_df[column], kde=True, ax=ax)
            ax.set_title("Original Data")
            st.pyplot(fig)

        with col3:
            quantile_transformation = QuantileTransformation(
                {column: transformations[transformation_options[column]]}
            )
            transformed_df = quantile_transformation.fit_transform(original_df)

            fig, ax = plt.subplots(figsize=(4, 2))
            sns.histplot(transformed_df[column], kde=True, ax=ax)
            ax.set_title("Transformed Data")
            st.pyplot(fig)

    st.markdown("""---""")

    # Apply button
    if st.button("Apply Transformations"):
        try:
            transformations_mapped = {
                col: transformations[transformation]
                for col, transformation in transformation_options.items()
            }

            quantile_transformation = QuantileTransformation(transformations_mapped)
            transformed_df = quantile_transformation.fit_transform(original_df)
            st.session_state["data"] = transformed_df

            config = st.session_state.get("config", {})
            config["quantile_transformation"] = {
                "transformation_options": quantile_transformation.transformation_options.copy(),
                "transformers": deepcopy(quantile_transformation.transformers),
            }
            st.session_state["config"] = config

            st.success("Transformations applied successfully!")

        except Exception as e:
            st.error("Error applying transformations: {}".format(e))

    # Next button
    next_button()
