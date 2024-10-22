import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import is_data_loaded
from cross.applications.styles import plot_remove_borders
from cross.transformations.preprocessing import Normalization
from cross.transformations.utils.dtypes import numerical_columns

from .normalization import NormalizationBase


class NormalizationPage(NormalizationBase):
    def show_page(self):
        st.title("Data Normalization")
        st.write("Choose a normalization technique for each column in your DataFrame.")

        if not is_data_loaded():
            return

        df, num_columns, original_df, target_column = self._initialize_data()

        transformation_options = {}

        self._display_transformation_options(
            df, num_columns, original_df, transformation_options
        )

        st.markdown("""---""")

        self._apply_transformations(df, transformation_options)

    def _initialize_data(self):
        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = numerical_columns(df)
        num_columns = [x for x in num_columns if x != target_column]

        return df, num_columns, original_df, target_column

    def _display_transformation_options(
        self, df, num_columns, original_df, transformation_options
    ):
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
                transformation_options[column] = selected_transformation

            with col2:
                self._display_distribution(original_df[column], "Original Data")

            with col3:
                transformed_df = self._apply_single_transformation(
                    original_df, column, transformation_options[column]
                )
                self._display_distribution(transformed_df[column], "Transformed Data")

    def _apply_single_transformation(self, df, column, transformation):
        normalization = Normalization({column: self.TRANSFORMATIONS[transformation]})
        transformed_df = normalization.fit_transform(df)
        return transformed_df

    def _display_distribution(self, column_data, title):
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.histplot(column_data, kde=True, ax=ax, color="#FF4C4B")
        ax.set_title(title)
        plot_remove_borders(ax)
        st.pyplot(fig)

    def _apply_transformations(self, df, transformation_options):
        if st.button("Add step"):
            try:
                transformations_mapped = {
                    col: self.TRANSFORMATIONS[transformation]
                    for col, transformation in transformation_options.items()
                    if self.TRANSFORMATIONS[transformation] != "none"
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
                st.error(f"Error applying transformations: {e}")
