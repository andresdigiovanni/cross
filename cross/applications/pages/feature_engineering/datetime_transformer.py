import streamlit as st

from cross.applications.components.check_is_data_loaded import is_data_loaded
from cross.core.feature_engineering.datetime_transformer import DateTimeTransformer
from cross.core.utils.dtypes import datetime_columns


class DateTimeTransformationPage:
    def show_page(self, name):
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

        # Select columns to transform
        st.subheader("Select Datetime Columns to Transform")
        datetime_columns_selected = st.multiselect(
            "Columns", options=datetime_cols, default=datetime_cols
        )

        st.markdown("""---""")

        # Show transformations
        st.subheader("Preview Transformations")
        for column in datetime_columns_selected:
            st.markdown(f"**Column: {column}**")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Original Data")
                st.dataframe(original_df[[column]].head())

            with col2:
                datetime_transformer = DateTimeTransformer(datetime_columns_selected)
                transformed_df = datetime_transformer.fit_transform(df)

                new_columns = list(
                    set(transformed_df.columns) - set(original_df.columns)
                )

                st.write("Transformed Data")
                st.dataframe(transformed_df[new_columns].drop_duplicates().head())

        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                datetime_transformer = DateTimeTransformer(datetime_columns_selected)
                transformed_df = datetime_transformer.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = datetime_transformer.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": name, "params": params})
                st.session_state["steps"] = steps

                st.success("Datetime columns transformed successfully!")

            except Exception as e:
                st.error(f"Error transforming datetime columns: {e}")
