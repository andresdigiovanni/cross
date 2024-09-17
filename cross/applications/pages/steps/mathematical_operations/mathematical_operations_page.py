import streamlit as st

from cross.applications.components import is_data_loaded
from cross.core.feature_engineering import MathematicalOperations
from cross.core.utils.dtypes import numerical_columns

from .mathematical_operations import MathematicalOperationsBase


class MathematicalOperationsPage(MathematicalOperationsBase):
    def show_page(self):
        st.title("Mathematical Operations")
        st.write("Select the mathematical operation for each pair of columns.")

        if not is_data_loaded():
            return

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = numerical_columns(df)
        num_columns = [x for x in num_columns if x != target_column]

        if "operations_options" not in st.session_state:
            st.session_state.operations_options = [
                (num_columns[0], num_columns[1], "add")
            ]

        def add_operation():
            st.session_state.operations_options.append(
                (num_columns[0], num_columns[1], "add")
            )

        for i, (col_a, col_b, operation) in enumerate(
            st.session_state.operations_options
        ):
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
                col_a = st.selectbox(
                    f"Select first column for operation {i + 1}",
                    num_columns,
                    index=num_columns.index(col_a) if col_a in num_columns else 0,
                    key=f"col_a_{i}",
                )
                col_b = st.selectbox(
                    f"Select second column for operation {i + 1}",
                    num_columns,
                    index=num_columns.index(col_b) if col_b in num_columns else 0,
                    key=f"col_b_{i}",
                )
                operation = st.selectbox(
                    f"Select operation {i + 1}",
                    self.operations.keys(),
                    index=list(self.operations.values()).index(operation)
                    if operation in self.operations.values()
                    else 0,
                    key=f"operation_{i}",
                )
                st.session_state.operations_options[i] = (
                    col_a,
                    col_b,
                    self.operations[operation],
                )

            with col2:
                st.write("Original Data")
                columns_to_select = list(set([col_a, col_b]))
                st.dataframe(original_df[columns_to_select].head())

            with col3:
                if self.operations[operation] != "none":
                    math_operations = MathematicalOperations(
                        [(col_a, col_b, self.operations[operation])]
                    )
                    transformed_df = math_operations.fit_transform(original_df)

                    new_column = f"{col_a}__{self.operations[operation]}__{col_b}"

                    st.write("Transformed Data")
                    st.dataframe(transformed_df[new_column].head())
                else:
                    st.write("No transformation applied")

        st.button("Add another operation", on_click=add_operation)
        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                math_operations = MathematicalOperations(
                    st.session_state.operations_options
                )
                transformed_df = math_operations.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = math_operations.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "MathematicalOperations", "params": params})
                st.session_state["steps"] = steps

                st.success("Mathematical operations applied successfully!")

            except Exception as e:
                st.error("Error applying operations: {}".format(e))
