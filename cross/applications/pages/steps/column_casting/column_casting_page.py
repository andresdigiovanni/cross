import streamlit as st

from cross.applications.components import is_data_loaded
from cross.core.preprocessing import CastColumns
from cross.core.utils.dtypes import (
    bool_columns,
    categorical_columns,
    datetime_columns,
    numerical_columns,
    timedelta_columns,
)


class ColumnCastingPage:
    def show_page(self, name):
        st.title("Column Casting")
        st.write("Modify column data types.")

        if not is_data_loaded():
            return

        df = st.session_state["data"]

        boolean_columns = bool_columns(df)
        cat_columns = categorical_columns(df)
        date_columns = datetime_columns(df)
        num_columns = numerical_columns(df)
        delta_columns = timedelta_columns(df)

        # Display data
        st.write(df.head())
        st.markdown("""---""")

        # Split window in 3 columns
        col1, col2, col3 = st.columns((1, 1, 1))

        # We create a dictionary to store the type selections for each column
        cast_options = {}
        original_types = {
            column: self._get_dtype(
                column,
                boolean_columns,
                cat_columns,
                date_columns,
                num_columns,
                delta_columns,
            )
            for column in df.columns
        }

        # Display columns selectors
        with col1:
            for i, column in enumerate(df.columns):
                if i % 3 == 0:
                    dtype = self._get_dtype(
                        column,
                        boolean_columns,
                        cat_columns,
                        date_columns,
                        num_columns,
                        delta_columns,
                    )
                    cast_options[column] = self._add_selectbox(column, dtype)

        with col2:
            for i, column in enumerate(df.columns):
                if i % 3 == 1:
                    dtype = self._get_dtype(
                        column,
                        boolean_columns,
                        cat_columns,
                        date_columns,
                        num_columns,
                        delta_columns,
                    )
                    cast_options[column] = self._add_selectbox(column, dtype)

        with col3:
            for i, column in enumerate(df.columns):
                if i % 3 == 2:
                    dtype = self._get_dtype(
                        column,
                        boolean_columns,
                        cat_columns,
                        date_columns,
                        num_columns,
                        delta_columns,
                    )
                    cast_options[column] = self._add_selectbox(column, dtype)

        st.markdown("""---""")

        # Convert button
        if st.button("Add step"):
            try:
                # only if there is a change
                cast_options = {
                    k: v for k, v in cast_options.items() if v != original_types[k]
                }

                cast_columns = CastColumns(cast_options)
                transformed_df = cast_columns.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = cast_columns.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": name, "params": params})
                st.session_state["steps"] = steps

                st.success("Columns successfully converted.")

            except Exception as e:
                st.error("Error converting columns: {}".format(e))

    def _add_selectbox(self, column, dtype):
        options = ["category", "number", "bool", "datetime", "timedelta"]

        return st.selectbox(
            "{}:".format(column),
            options=options,
            index=options.index(dtype),
            key=column,
        )

    def _get_dtype(
        self,
        column,
        boolean_columns,
        cat_columns,
        date_columns,
        num_columns,
        delta_columns,
    ):
        if column in boolean_columns:
            return "bool"

        if column in date_columns:
            return "datetime"

        elif column in num_columns:
            return "number"

        if column in delta_columns:
            return "timedelta"

        return "category"
