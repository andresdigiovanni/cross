import streamlit as st

from cross.applications.components import is_data_loaded
from cross.transformations import MissingValuesIndicator


class MissingValuesIndicatorPage:
    def show_page(self):
        st.title("Missing Values Indicator")
        st.write(
            "Identify and flag missing values in your DataFrame. "
            "This tool allows you to create missing value indicators for selected columns."
        )

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        original_df = df.copy()
        missing_values = df.isnull().sum()

        features = self._display_missing_values_indicator(
            original_df,
            missing_values,
        )

        st.markdown("""---""")
        self._apply_transformation_button(df, features)

    def _display_missing_values_indicator(self, df, missing_values):
        features = set()

        for column in df.columns:
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(column)
                st.write(f"Missing values: {missing_values[column]}")
                is_selected = st.checkbox(
                    "Missing value indicator", key=f"{column}_missing"
                )

            with col2:
                st.write("Original Data")
                st.dataframe(df[[column]].head())

            with col3:
                if is_selected:
                    indicator = MissingValuesIndicator(features=[column])
                    transformed_df = indicator.fit_transform(df)
                    new_columns = list(set(transformed_df.columns) - set(df.columns))

                    st.write("Transformed Data")
                    st.dataframe(transformed_df[new_columns].head())

                    features.add(column)

                else:
                    st.write("No transformation applied")
                    if column in features:
                        features.remove(column)

        return list(features)

    def _apply_transformation_button(self, df, features):
        if st.button("Add step"):
            try:
                indicator = MissingValuesIndicator(features=features)
                transformed_df = indicator.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = indicator.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "MissingValuesIndicator", "params": params})
                st.session_state["steps"] = steps

                st.success("Missing values indicator applied successfully!")

            except Exception as e:
                st.error(f"Error applying indicator: {e}")
