import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import next_button
from cross.core.preprocessing.normalization import Normalization
from cross.core.utils.dtypes import numerical_columns


class NormalizationPage:
    def show_page(self):
        st.title("Data Normalization")
        st.write("Choose a normalization technique for each column in your DataFrame.")

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
            return

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = numerical_columns(df)

        # Actions for each column
        transformations = {
            "Do nothing": "none",
            "Normalize (L1)": "l1",
            "Normalize (L2)": "l2",
        }

        transformation_options = {}

        for column in num_columns:
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
                normalization = Normalization(
                    {column: transformations[transformation_options[column]]}
                )
                transformed_df = normalization.fit_transform(original_df)

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

                normalization = Normalization(transformations_mapped)
                transformed_df = normalization.fit_transform(original_df)
                st.session_state["data"] = transformed_df

                config = st.session_state.get("config", {})
                config["normalization"] = normalization.get_params()
                st.session_state["config"] = config

                st.success("Transformations applied successfully!")

            except Exception as e:
                st.error("Error applying transformations: {}".format(e))

        # Next button
        next_button()
