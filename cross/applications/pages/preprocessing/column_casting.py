import streamlit as st

from cross.applications.components import next_button
from cross.core.preprocessing import CastColumns
from cross.core.utils.dtypes import (
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

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
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

        # Convert button
        st.markdown("""---""")
        if st.button("Cast columns"):
            try:
                cast_columns = CastColumns(cast_options)
                transformed_df = cast_columns.fit_transform(df)
                st.session_state["data"] = transformed_df

                config = st.session_state.get("config", {})
                config["column_casting"] = cast_columns.get_params()
                st.session_state["config"] = config

                st.success("Columns successfully converted.")

            except Exception as e:
                st.error("Error converting columns: {}".format(e))

        # Next button
        next_button()

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
