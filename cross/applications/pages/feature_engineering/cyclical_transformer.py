import streamlit as st

from cross.applications.components import is_data_loaded
from cross.core.feature_engineering.cyclical_features_transformer import (
    CyclicalFeaturesTransformer,
)


class CyclicalFeaturesTransformationPage:
    def show_page(self, name):
        st.title("Cyclical Features Transformation")
        st.write(
            "Transform cyclical features in your DataFrame. "
            "This will extract the sine and cosine components for the selected columns."
        )

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        original_df = df.copy()

        cyclical_columns = st.multiselect(
            "Select Cyclical Columns", options=df.columns.tolist(), default=[]
        )

        columns_periods = {}
        for column in cyclical_columns:
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                default_period = self.get_default_period(original_df, column)
                period = st.number_input(
                    f"Period for {column}",
                    min_value=1,
                    value=default_period,
                    step=1,
                    key=f"{column}_period",
                )
                columns_periods[column] = period

            with col2:
                st.write("Original Data")
                st.dataframe(original_df[[column]].head())

            with col3:
                cyclical_transformer = CyclicalFeaturesTransformer(columns_periods)
                transformed_df = cyclical_transformer.fit_transform(df)

                new_columns = [f"{column}_sin", f"{column}_cos"]

                st.write("Transformed Data")
                st.dataframe(transformed_df[new_columns].head())

        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                cyclical_transformer = CyclicalFeaturesTransformer(columns_periods)
                transformed_df = cyclical_transformer.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = cyclical_transformer.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": name, "params": params})
                st.session_state["steps"] = steps

                st.success("Cyclical features transformed successfully!")

            except Exception as e:
                st.error(f"Error transforming cyclical features: {e}")

    def get_default_period(self, df, column):
        unique_values = df[column].dropna().unique()

        if column.lower().endswith("month"):
            return 12

        elif column.lower().endswith("day"):
            return 31

        elif column.lower().endswith("hour"):
            return 24

        elif column.lower().endswith("minute") or column.lower().endswith("second"):
            return 60

        else:
            return len(unique_values)
