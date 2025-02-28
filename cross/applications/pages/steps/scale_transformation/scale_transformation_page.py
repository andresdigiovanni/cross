import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import is_data_loaded
from cross.applications.styles import plot_remove_borders
from cross.transformations import ScaleTransformation
from cross.transformations.utils.dtypes import numerical_columns

from .scale_transformation import ScaleTransformationsBase


class ScaleTransformationsPage(ScaleTransformationsBase):
    def show_page(self):
        st.title("Scale Transformations")
        st.write("Apply various scaling transformations to your DataFrame columns.")

        if not is_data_loaded():
            return

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = [col for col in numerical_columns(df) if col != target_column]

        transformation_options = {}
        quantile_ranges = {}

        for column in num_columns:
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(column)
                selected_transformation = st.selectbox(
                    f"Select transformation for {column}",
                    self.TRANSFORMATIONS.keys(),
                    key=column,
                )
                transformation_options[column] = self.TRANSFORMATIONS[
                    selected_transformation
                ]

                if transformation_options[column] == "robust":
                    quantile_range = st.slider(
                        f"Select quantile range for {column}",
                        min_value=1.0,
                        max_value=99.0,
                        value=(25.0, 75.0),
                        key=f"{column}_quantile_range",
                    )
                    quantile_ranges[column] = quantile_range

            with col2:
                self._plot_column_data(original_df[column], "Original Data")

            with col3:
                transformed_df = self._apply_transformation(
                    original_df.copy(),
                    column,
                    transformation_options[column],
                    quantile_ranges,
                )
                self._plot_column_data(transformed_df[column], "Transformed Data")

        st.markdown("""---""")
        self._apply_scale_transformations(df, transformation_options, quantile_ranges)

    def _plot_column_data(self, column_data, title):
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.histplot(column_data, kde=True, ax=ax, color="#FF4C4B")
        ax.set_title(title)
        plot_remove_borders(ax)
        st.pyplot(fig)

    def _apply_transformation(self, df, column, transformation, quantile_ranges):
        scale_transformation = ScaleTransformation(
            {column: transformation}, quantile_ranges
        )
        return scale_transformation.fit_transform(df)

    def _apply_scale_transformations(self, df, transformation_options, quantile_ranges):
        if st.button("Add step"):
            try:
                valid_transformations = {
                    col: transformation
                    for col, transformation in transformation_options.items()
                    if transformation != "none"
                }

                scale_transformation = ScaleTransformation(
                    valid_transformations, quantile_ranges
                )
                transformed_df = scale_transformation.fit_transform(df)
                st.session_state["data"] = transformed_df

                # Update session state with transformations
                params = scale_transformation.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "ScaleTransformation", "params": params})
                st.session_state["steps"] = steps

                st.success("Transformations applied successfully!")
            except Exception as e:
                st.error(f"Error applying transformations: {e}")
