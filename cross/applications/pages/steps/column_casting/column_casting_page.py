import streamlit as st

from cross.applications.components import is_data_loaded
from cross.transformations import CastColumns
from cross.transformations.utils.dtypes import (
    bool_columns,
    categorical_columns,
    datetime_columns,
    numerical_columns,
    timedelta_columns,
)


class ColumnCastingPage:
    def show_page(self):
        st.title("Column Casting")
        st.write("Modify column data types.")

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        target_column = st.session_state.get("config", {}).get("target_column", None)

        # Identify columns by type
        type_columns = {
            "bool": bool_columns(df),
            "category": categorical_columns(df),
            "datetime": datetime_columns(df),
            "number": numerical_columns(df),
            "timedelta": timedelta_columns(df),
        }

        # Exclude target column
        columns = [col for col in df.columns if col != target_column]

        # Create dictionary for current types and cast options
        original_types = {col: self._get_dtype(col, type_columns) for col in columns}
        cast_options = {}

        # Display data and columns
        st.write(df.head())
        st.markdown("""---""")
        col1, col2, col3 = st.columns(3)

        # Distribute column selectors across three columns
        for i, column in enumerate(columns):
            col = [col1, col2, col3][i % 3]
            with col:
                dtype = original_types[column]
                cast_options[column] = self._add_selectbox(column, dtype)

        st.markdown("""---""")

        # Convert button
        if st.button("Add step"):
            self._apply_casting(df, original_types, cast_options)

    def _get_dtype(self, column, type_columns):
        for dtype, cols in type_columns.items():
            if column in cols:
                return dtype
        return "category"

    def _add_selectbox(self, column, dtype):
        options = ["category", "number", "bool", "datetime", "timedelta"]
        return st.selectbox(
            f"{column}:", options, index=options.index(dtype), key=column
        )

    def _apply_casting(self, df, original_types, cast_options):
        try:
            # Apply only if there is a type change
            cast_options = {
                col: dtype
                for col, dtype in cast_options.items()
                if dtype != original_types[col]
            }
            if not cast_options:
                st.warning("No changes were made to column types.")
                return

            # Apply the transformations
            cast_columns = CastColumns(cast_options)
            transformed_df = cast_columns.fit_transform(df)
            st.session_state["data"] = transformed_df

            # Save the step
            steps = st.session_state.get("steps", [])
            steps.append({"name": "CastColumns", "params": cast_columns.get_params()})
            st.session_state["steps"] = steps

            st.success("Columns successfully converted.")
        except Exception as e:
            st.error(f"Error converting columns: {e}")
