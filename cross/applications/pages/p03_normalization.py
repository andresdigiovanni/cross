from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import next_button
from cross.normalization import normalize_column, normalize_data


def show_page():
    st.title("Data Normalization")
    st.write("Choose a normalization technique for each column in your DataFrame.")

    if "data" not in st.session_state:
        st.warning("No data loaded. Please load a DataFrame.")
        return

    df = st.session_state["data"]
    original_df = df.copy()

    # Actions for each column
    transformations = {
        "Do nothing": "none",
        "Min-Max Scaling": "min_max_scaling",
        "Standardization": "standardization",
        "Robust Scaling": "robust_scaling",
        "Normalization": "normalization",
        "MaxAbs Scaling": "max_abs_scaling",
        "Quantile Transformation (Uniform)": "quantile_uniform",
        "Quantile Transformation (Normal)": "quantile_normal",
        "Log Transformation": "log",
        "Exponential Transformation": "exponential",
        "Yeo-Johnson Transformation": "yeo_johnson",
    }

    transformation_options = {}
    transformation_results = {}

    for column in df.columns:
        st.markdown("""---""")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(column)
            transformation_options[column] = st.selectbox(
                f"Select transformation for {column}",
                transformations.keys(),
                key=column,
            )

        with col2:
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.histplot(original_df[column], kde=True, ax=ax)
            ax.set_title("Original Data")
            st.pyplot(fig)

        with col3:
            transformed_df, _ = normalize_column(
                original_df.copy(),
                column,
                transformations[transformation_options[column]],
            )
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.histplot(transformed_df[column], kde=True, ax=ax)
            ax.set_title("Transformed Data")
            st.pyplot(fig)

            transformation_results[column] = transformed_df[column]

    st.markdown("""---""")

    # Apply button
    if st.button("Apply Transformations"):
        try:
            transformations_mapped = {
                col: transformation_options[transformation]
                for col, transformation in transformation_options.items()
            }
            df, transformers = normalize_data(df, column, transformations_mapped)
            st.session_state["data"] = df

            config = st.session_state.get("config", {})
            config["normalization"] = transformations_mapped.copy()
            config["normalization_transformers"] = deepcopy(transformers)
            st.session_state["config"] = config

            st.success("Transformations applied successfully!")

        except Exception as e:
            st.error("Error applying transformations: {}".format(e))

    # Next button
    next_button()
