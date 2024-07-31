import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components.check_is_data_loaded import is_data_loaded
from cross.core.preprocessing.non_linear_transformation import NonLinearTransformation
from cross.core.utils.dtypes import numerical_columns


class NonLinearTransformationPage:
    def show_page(self, name):
        st.title("Non-linear Transformations")
        st.write("Apply various non-linear transformations to your DataFrame columns.")

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = numerical_columns(df)

        # Actions for each column
        transformations = {
            "Do nothing": "none",
            "Log Transformation": "log",
            "Exponential Transformation": "exponential",
            "Yeo-Johnson Transformation": "yeo_johnson",
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
                non_linear_transformation = NonLinearTransformation(
                    {column: transformations[transformation_options[column]]}
                )
                transformed_df = non_linear_transformation.fit_transform(original_df)

                fig, ax = plt.subplots(figsize=(4, 2))
                sns.histplot(transformed_df[column], kde=True, ax=ax)
                ax.set_title("Transformed Data")
                st.pyplot(fig)

        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                transformations_mapped = {
                    col: transformations[transformation]
                    for col, transformation in transformation_options.items()
                }

                non_linear_transformation = NonLinearTransformation(
                    transformations_mapped
                )
                transformed_df = non_linear_transformation.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = non_linear_transformation.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": name, "params": params})
                st.session_state["steps"] = steps

                st.success("Transformations applied successfully!")

            except Exception as e:
                st.error("Error applying transformations: {}".format(e))
