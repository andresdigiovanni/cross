import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import is_data_loaded
from cross.applications.styles import plot_remove_borders
from cross.core.preprocessing import Normalization
from cross.core.utils.dtypes import numerical_columns

from .normalization import NormalizationBase


class NormalizationPage(NormalizationBase):
    def show_page(self):
        st.title("Data Normalization")
        st.write("Choose a normalization technique for each column in your DataFrame.")

        if not is_data_loaded():
            return

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = numerical_columns(df)
        num_columns = [x for x in num_columns if x != target_column]

        transformation_options = {}

        for column in num_columns:
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(column)
                selected_transformation = st.selectbox(
                    f"Select transformation for {column}",
                    self.transformations.keys(),
                    key=column,
                )
                transformation_options[column] = selected_transformation

            with col2:
                fig, ax = plt.subplots(figsize=(4, 2))
                sns.histplot(original_df[column], kde=True, ax=ax, color="#FF4C4B")

                ax.set_title("Original Data")
                plot_remove_borders(ax)

                st.pyplot(fig)

            with col3:
                normalization = Normalization(
                    {column: self.transformations[transformation_options[column]]}
                )
                transformed_df = normalization.fit_transform(original_df)

                fig, ax = plt.subplots(figsize=(4, 2))
                sns.histplot(transformed_df[column], kde=True, ax=ax, color="#FF4C4B")

                ax.set_title("Transformed Data")
                plot_remove_borders(ax)

                st.pyplot(fig)

        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                transformations_mapped = {
                    col: self.transformations[transformation]
                    for col, transformation in transformation_options.items()
                }

                normalization = Normalization(transformations_mapped)
                transformed_df = normalization.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = normalization.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "Normalization", "params": params})
                st.session_state["steps"] = steps

                st.success("Transformations applied successfully!")

            except Exception as e:
                st.error("Error applying transformations: {}".format(e))
