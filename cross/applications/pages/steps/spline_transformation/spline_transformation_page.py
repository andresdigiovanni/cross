import streamlit as st

from cross.applications.components import is_data_loaded
from cross.transformations import SplineTransformation
from cross.transformations.utils.dtypes import numerical_columns

from .spline_transformation import SplineTransformationsBase


class SplineTransformationsPage(SplineTransformationsBase):
    def show_page(self):
        st.title("Spline Transformations")
        st.write(
            "Select and apply spline transformations to each numerical column of your dataset."
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
                selected_transformation = self.TRANSFORMATIONS[selected_transformation]

                if selected_transformation != "none":
                    degree = st.slider(
                        f"Select the degree for {column}",
                        min_value=1,
                        max_value=8,
                        value=3,
                        key=f"{column}_degree",
                    )
                    n_knots = st.slider(
                        f"Select number of knots for {column}",
                        min_value=3,
                        max_value=20,
                        value=5,
                        key=f"{column}_knots",
                    )
                    transformation_options[column] = {
                        "degree": degree,
                        "n_knots": n_knots,
                    }
                else:
                    if column in transformation_options:
                        del transformation_options[column]

            with col2:
                st.write("Original Data")
                st.dataframe(df[[column]].head())

            with col3:
                if column in transformation_options:
                    transformed_df = self._apply_transformation(
                        df.copy(), column, transformation_options[column]
                    )

                    st.write("Transformed Data")
                    st.dataframe(transformed_df.drop(columns=df.columns).head())
                else:
                    st.write("No spline applied")

        st.markdown("""---""")
        self._apply_spline_transformations(df, transformation_options)

    def _apply_transformation(self, df, column, transformation):
        spline_transformation = SplineTransformation({column: transformation})
        return spline_transformation.fit_transform(df)

    def _apply_spline_transformations(self, df, transformation_options):
        if st.button("Add step"):
            try:
                spline_transformation = SplineTransformation(transformation_options)
                transformed_df = spline_transformation.fit_transform(df)
                st.session_state["data"] = transformed_df

                # Update session state with transformations
                params = spline_transformation.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "SplineTransformation", "params": params})
                st.session_state["steps"] = steps

                st.success("Transformations applied successfully!")

            except Exception as e:
                st.error(f"Error applying transformations: {e}")
