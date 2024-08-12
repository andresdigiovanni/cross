import streamlit as st

from .remove_duplicates import RemoveDuplicatesBase


class RemoveDuplicatesEdit(RemoveDuplicatesBase):
    def show_component(self, params):
        subset = params["subset"]
        keep = params["keep"]
        reverse_options = {v: k for k, v in self.keep_options.items()}
        n_cols = 4

        st.write("Handle duplicates method: **{}**".format(reverse_options[keep]))
        st.write("**Columns:**")

        cols = st.columns((1,) * n_cols)
        columns_list = [[] for _ in range(n_cols)]

        for i, col in enumerate(subset):
            columns_list[i % n_cols].append(col)

        for col, column_list in zip(cols, columns_list):
            with col:
                for column in column_list:
                    st.write(column)
