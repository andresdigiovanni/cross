import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import is_data_loaded
from cross.applications.styles import plot_remove_borders
from cross.transformations import QuantileTransformation
from cross.transformations.utils.dtypes import numerical_columns

from .quantile_transformation import QuantileTransformationsBase


class QuantileTransformationsPage(QuantileTransformationsBase):
    def show_page(self):
        st.title("Quantile Transformations")
        st.write(
            "Select and apply quantile transformations (Uniform or Normal) to each column of your dataset."
        )

        if not is_data_loaded():
            return

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        df = st.session_state["data"]
        num_columns = [col for col in numerical_columns(df) if col != target_column]

        transformation_options = {}

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

            with col2:
                self._plot_column_data(df[column], "Original Data")

            with col3:
                transformed_df = self._apply_transformation(
                    df.copy(), column, transformation_options[column]
                )
                self._plot_column_data(transformed_df[column], "Transformed Data")

        st.markdown("""---""")
        self._apply_quantile_transformations(df, transformation_options)

    def _plot_column_data(self, column_data, title):
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.histplot(column_data, kde=True, ax=ax, color="#FF4C4B")
        ax.set_title(title)
        plot_remove_borders(ax)
        st.pyplot(fig)

    def _apply_transformation(self, df, column, transformation):
        quantile_transformation = QuantileTransformation({column: transformation})
        return quantile_transformation.fit_transform(df)

    def _apply_quantile_transformations(self, df, transformation_options):
        if st.button("Add step"):
            try:
                valid_transformations = {
                    col: transformation
                    for col, transformation in transformation_options.items()
                    if transformation != "none"
                }

                quantile_transformation = QuantileTransformation(valid_transformations)
                transformed_df = quantile_transformation.fit_transform(df)
                st.session_state["data"] = transformed_df

                # Update session state with transformations
                params = quantile_transformation.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "QuantileTransformation", "params": params})
                st.session_state["steps"] = steps

                # Update config with quantile transformations
                config = st.session_state.get("config", {})
                config["quantile_transformation"] = params
                st.session_state["config"] = config

                st.success("Transformations applied successfully!")
            except Exception as e:
                st.error(f"Error applying transformations: {e}")
