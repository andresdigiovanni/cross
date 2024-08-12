import streamlit as st


class ColumnSelectionEdit:
    def show_component(self, params):
        columns = params["columns"]
        n_cols = 4

        st.write("**Columns:**")

        cols = st.columns((1,) * n_cols)
        columns_list = [[] for _ in range(n_cols)]

        for i, col in enumerate(columns):
            columns_list[i % n_cols].append(col)

        for col, column_list in zip(cols, columns_list):
            with col:
                for column in column_list:
                    st.write(column)
