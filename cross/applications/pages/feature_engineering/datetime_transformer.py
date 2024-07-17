import streamlit as st

from cross.applications.components import next_button
from cross.feature_engineering.datetime_transformer import DateTimeTransformer
from cross.load_data.dtypes import datetime_columns


class DateTimeTransformationPage:
    def show_page(self):
        st.title("Datetime Transformation")
        st.write(
            "Transform datetime columns in your DataFrame. "
            "This will extract the year, month, day, hour, minute, and second components from the selected columns."
        )

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
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
        if st.button("Apply"):
            try:
                datetime_transformer = DateTimeTransformer(datetime_columns_selected)
                df = datetime_transformer.fit_transform(df)
                st.session_state["data"] = df

                config = st.session_state.get("config", {})
                config["datetime_transformation"] = {
                    "datetime_columns": datetime_columns_selected
                }
                st.session_state["config"] = config

                st.success("Datetime columns transformed successfully!")

            except Exception as e:
                st.error(f"Error transforming datetime columns: {e}")

        # Next button
        next_button()
