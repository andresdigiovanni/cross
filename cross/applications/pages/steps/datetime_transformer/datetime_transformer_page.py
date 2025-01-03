import streamlit as st

from cross.applications.components import is_data_loaded
from cross.transformations import DateTimeTransformer
from cross.transformations.utils.dtypes import datetime_columns


class DateTimeTransformationPage:
    def show_page(self):
        st.title("Datetime Transformation")
        st.write(
            "Transform datetime columns in your DataFrame. "
            "This will extract the year, month, day, hour, minute, and second components from the selected columns."
        )

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        original_df = df.copy()
        datetime_cols = datetime_columns(df)

        self._display_column_selector(datetime_cols)
        datetime_columns_selected = st.session_state.get(
            "datetime_columns_selected", datetime_cols
        )

        st.markdown("""---""")
        self._preview_transformations(original_df, datetime_columns_selected)

        st.markdown("""---""")
        self._apply_transformation_button(df, datetime_columns_selected)

    def _display_column_selector(self, datetime_cols):
        st.subheader("Select Datetime Columns to Transform")
        datetime_columns_selected = st.multiselect(
            "Columns", options=datetime_cols, default=datetime_cols
        )
        st.session_state["datetime_columns_selected"] = datetime_columns_selected

    def _preview_transformations(self, original_df, datetime_columns_selected):
        st.subheader("Preview Transformations")
        for column in datetime_columns_selected:
            st.markdown(f"**Column: {column}**")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Original Data")
                st.dataframe(original_df[[column]].head())

            with col2:
                datetime_transformer = DateTimeTransformer(datetime_columns_selected)
                transformed_df = datetime_transformer.fit_transform(original_df)
                new_columns = list(
                    set(transformed_df.columns) - set(original_df.columns)
                )

                st.write("Transformed Data")
                st.dataframe(transformed_df[new_columns].drop_duplicates().head())

    def _apply_transformation_button(self, df, datetime_columns_selected):
        if st.button("Add step"):
            try:
                datetime_transformer = DateTimeTransformer(datetime_columns_selected)
                transformed_df = datetime_transformer.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = datetime_transformer.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "DateTimeTransformer", "params": params})
                st.session_state["steps"] = steps

                st.success("Datetime columns transformed successfully!")

            except Exception as e:
                st.error(f"Error transforming datetime columns: {e}")
